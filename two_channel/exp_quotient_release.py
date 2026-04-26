import argparse,json,os,time,math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset,DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from .quotient_release import QuotientRelease,reparam,kl_iso,info_nce
from .gradient_reversal import grad_reverse
from .adjacency_builder_v2 import get_layer_block

class WikiC4Stream(IterableDataset):
    def __init__(self,tok,seq_len,mix=0.8,seed=0,c4_max_lines=200000):
        self.tok=tok
        self.seq_len=seq_len
        self.mix=mix
        self.seed=seed
        self.c4_max=c4_max_lines
    def __iter__(self):
        from datasets import load_dataset
        wt=load_dataset("wikitext","wikitext-103-raw-v1",split="train",streaming=True)
        c4=load_dataset("allenai/c4","en",split="train",streaming=True)
        rng=torch.Generator().manual_seed(self.seed)
        wt_iter=iter(wt)
        c4_iter=iter(c4)
        c4_seen=0
        while True:
            use_c4=(torch.rand(1,generator=rng).item()>self.mix) and (c4_seen<self.c4_max)
            try:
                row=next(c4_iter) if use_c4 else next(wt_iter)
            except StopIteration:
                if use_c4: c4_iter=iter(c4)
                else: wt_iter=iter(wt)
                continue
            if use_c4: c4_seen+=1
            txt=row.get("text","").strip()
            if len(txt)<80: continue
            ids=self.tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
            if len(ids)>=self.seq_len:
                s=int(torch.randint(0,len(ids)-self.seq_len+1,(1,),generator=rng).item())
                yield torch.tensor(ids[s:s+self.seq_len],dtype=torch.long)

def collate(batch):
    return torch.stack(batch,dim=0)

class FrozenLM:
    def __init__(self,model_name,layer,dev,dtype):
        self.tok=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=dtype).to(dev).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.layer=layer
        self.dev=dev
        self.blk=get_layer_block(self.model,layer)
        self.d=self.model.config.hidden_size
        self.vocab=self.model.config.vocab_size
        self._cap=[None]
        self._inj=None
    def hidden_seq(self,ids):
        cap=[None]
        def hk(m,i,o,c=cap):
            c[0]=(o[0] if isinstance(o,tuple) else o).detach()
        h=self.blk.register_forward_hook(hk)
        with torch.no_grad():
            self.model(ids)
        h.remove()
        return cap[0].float()
    def logits_with_replaced_hidden(self,ids,h_new):
        def inj(m,i,o,hn=h_new):
            oo=o[0] if isinstance(o,tuple) else o
            oo=hn.to(oo.dtype)
            if isinstance(o,tuple):return(oo,)+o[1:]
            return oo
        h=self.blk.register_forward_hook(inj)
        out=self.model(ids)
        h.remove()
        return out.logits.float()

def kl_pq(logp,logq):
    p=logp.exp()
    return (p*(logp-logq)).sum(-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--layer",type=int,required=True)
    ap.add_argument("--r",type=int,default=64)
    ap.add_argument("--beta",type=float,default=1e-3)
    ap.add_argument("--gamma",type=float,default=0.1)
    ap.add_argument("--sigma_rel",type=float,default=0.2)
    ap.add_argument("--seq_len",type=int,default=64)
    ap.add_argument("--H_horizon",type=int,default=16)
    ap.add_argument("--batch_size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--steps",type=int,default=100000)
    ap.add_argument("--warmup",type=int,default=2000)
    ap.add_argument("--dtype",default="float32")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--out_dir",default="artifacts/quotient_release")
    ap.add_argument("--log_every",type=int,default=200)
    ap.add_argument("--ckpt_every",type=int,default=10000)
    a=ap.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    torch.manual_seed(a.seed)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    dtype=getattr(torch,a.dtype)
    print(f"loading {a.model} layer {a.layer} on {dev} {a.dtype}")
    flm=FrozenLM(a.model,a.layer,dev,dtype)
    print(f"d={flm.d} vocab={flm.vocab}")
    qr=QuotientRelease(flm.d,a.r).to(dev).to(torch.float32)
    opt=torch.optim.AdamW(qr.parameters(),lr=a.lr,weight_decay=0.01)
    def lr_at(step):
        if step<a.warmup: return step/a.warmup
        prog=(step-a.warmup)/max(1,a.steps-a.warmup)
        return 0.5*(1+math.cos(math.pi*min(1.0,prog)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_at)
    ds=WikiC4Stream(flm.tok,a.seq_len,seed=a.seed)
    loader=DataLoader(ds,batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    it=iter(loader)
    t0=time.time()
    losses=[]
    slug=f"{a.model.replace('/','_')}_L{a.layer}_r{a.r}_b{a.beta:.0e}_g{a.gamma}_s{a.sigma_rel}_seed{a.seed}"
    log_path=os.path.join(a.out_dir,f"{slug}.log.json")
    log_data=[]
    for step in range(a.steps):
        ids=next(it).to(dev)
        with torch.no_grad():
            h_clean=flm.hidden_seq(ids)
        h_clean=h_clean.requires_grad_(False)
        out=qr(h_clean,sigma_rel=a.sigma_rel,grl=1.0)
        h_hat=out["h_hat"]
        logits_hat=flm.logits_with_replaced_hidden(ids,h_hat)
        with torch.no_grad():
            logits_clean=flm.logits_with_replaced_hidden(ids,h_clean)
        T=logits_hat.size(1)
        H=min(a.H_horizon,T)
        lp_clean=F.log_softmax(logits_clean[:,-H:].float(),dim=-1).detach()
        lp_hat=F.log_softmax(logits_hat[:,-H:].float(),dim=-1)
        L_util=kl_pq(lp_clean,lp_hat).mean()
        L_ib=kl_iso(out["mu"],out["ls"])
        L_ret=info_nce(out["u"],out["v"])
        L=L_util+a.beta*L_ib+a.gamma*L_ret
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(qr.parameters(),1.0)
        opt.step()
        sched.step()
        if step%a.log_every==0:
            log_data.append(dict(step=step,L=L.item(),L_util=L_util.item(),L_ib=L_ib.item(),L_ret=L_ret.item(),lr=sched.get_last_lr()[0]))
            print(f"step {step} L={L.item():.4f} util={L_util.item():.4f} ib={L_ib.item():.4f} ret={L_ret.item():.4f}")
            with open(log_path,"w") as f:json.dump(log_data,f,indent=2)
        if step>0 and step%a.ckpt_every==0:
            torch.save(qr.state_dict(),os.path.join(a.out_dir,f"{slug}.step{step}.pt"))
    final={"slug":slug,"model":a.model,"layer":a.layer,"r":a.r,"beta":a.beta,"gamma":a.gamma,"sigma_rel":a.sigma_rel,"steps":a.steps,"elapsed_s":time.time()-t0,"log":log_data}
    with open(os.path.join(a.out_dir,f"{slug}.json"),"w") as f:
        json.dump(final,f,indent=2)
    torch.save(qr.state_dict(),os.path.join(a.out_dir,f"{slug}.final.pt"))
    print(f"saved {slug}")

if __name__=="__main__":
    main()

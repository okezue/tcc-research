import argparse,json,os,time,math,glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset,DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from .sequence_inverter import SeqInv,beam_search
from .quotient_release import QuotientRelease
from .adjacency_builder_v2 import get_layer_block

class WikiTextDS(IterableDataset):
    _cache=None
    @classmethod
    def get_pretok(cls,tok,seq_lens,n_samples):
        if cls._cache is not None:return cls._cache
        from datasets import load_dataset
        ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
        max_sl=max(seq_lens)
        out=[]
        for row in ds:
            txt=row["text"].strip()
            if len(txt)<80:continue
            ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
            if len(ids)>=max_sl:
                out.append(ids)
                if len(out)>=n_samples:break
        cls._cache=out
        return out
    def __init__(self,tok,seq_lens,n_samples=200000,seed=0):
        self.tok=tok
        self.seq_lens=seq_lens
        self.seed=seed
        self.cache=self.get_pretok(tok,seq_lens,n_samples)
    def __iter__(self):
        rng=torch.Generator().manual_seed(self.seed)
        N=len(self.cache)
        while True:
            i=int(torch.randint(0,N,(1,),generator=rng).item())
            ids=self.cache[i]
            sl=self.seq_lens[int(torch.randint(0,len(self.seq_lens),(1,),generator=rng).item())]
            if len(ids)<sl:continue
            s=int(torch.randint(0,len(ids)-sl+1,(1,),generator=rng).item())
            yield torch.tensor(ids[s:s+sl],dtype=torch.long),sl

def collate(batch):
    ids=[b[0] for b in batch]
    lens=[b[1] for b in batch]
    L=max(len(x) for x in ids)
    out=torch.full((len(ids),L),0,dtype=torch.long)
    mask=torch.zeros(len(ids),L,dtype=torch.bool)
    for i,x in enumerate(ids):
        out[i,:len(x)]=x
        mask[i,:len(x)]=True
    return out,mask,torch.tensor(lens)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target_model",required=True)
    ap.add_argument("--target_layer",type=int,required=True)
    ap.add_argument("--quotient_ckpt",required=True)
    ap.add_argument("--r",type=int,required=True)
    ap.add_argument("--sigma_rel",type=float,default=0.2)
    ap.add_argument("--mech_id",type=int,default=0)
    ap.add_argument("--seq_lens",type=str,default="8,16,32,64")
    ap.add_argument("--max_T",type=int,default=64)
    ap.add_argument("--dm",type=int,default=512)
    ap.add_argument("--nhead",type=int,default=8)
    ap.add_argument("--enc_layers",type=int,default=6)
    ap.add_argument("--dec_layers",type=int,default=6)
    ap.add_argument("--ff",type=int,default=2048)
    ap.add_argument("--drop",type=float,default=0.1)
    ap.add_argument("--batch_size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--steps",type=int,default=100000)
    ap.add_argument("--warmup",type=int,default=2000)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--target_dtype",default="bfloat16")
    ap.add_argument("--out_dir",default="artifacts/sequence_inverter")
    ap.add_argument("--log_every",type=int,default=500)
    ap.add_argument("--ckpt_every",type=int,default=25000)
    a=ap.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    torch.manual_seed(a.seed)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tdtype=getattr(torch,a.target_dtype)
    tok=AutoTokenizer.from_pretrained(a.target_model)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    target=AutoModelForCausalLM.from_pretrained(a.target_model,torch_dtype=tdtype).to(dev).eval()
    for p in target.parameters():p.requires_grad_(False)
    blk=get_layer_block(target,a.target_layer)
    d=target.config.hidden_size
    qr=QuotientRelease(d,a.r).to(dev).to(torch.float32).eval()
    qr.load_state_dict(torch.load(a.quotient_ckpt,map_location=dev))
    for p in qr.parameters():p.requires_grad_(False)
    inv=SeqInv(r=a.r,vocab=tok.vocab_size,dm=a.dm,nhead=a.nhead,enc_layers=a.enc_layers,dec_layers=a.dec_layers,ff=a.ff,drop=a.drop,max_T=a.max_T).to(dev)
    n_params=sum(p.numel() for p in inv.parameters())
    print(f"target d={d} vocab={tok.vocab_size} inverter_params={n_params/1e6:.1f}M")
    opt=torch.optim.AdamW(inv.parameters(),lr=a.lr,weight_decay=0.01)
    def lr_at(step):
        if step<a.warmup:return step/a.warmup
        prog=(step-a.warmup)/max(1,a.steps-a.warmup)
        return 0.5*(1+math.cos(math.pi*min(1.0,prog)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_at)
    seq_lens=[int(x) for x in a.seq_lens.split(",")]
    ds=WikiTextDS(tok,seq_lens,seed=a.seed)
    loader=DataLoader(ds,batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    it=iter(loader)
    slug=f"{a.target_model.replace('/','_')}_L{a.target_layer}_r{a.r}_s{a.sigma_rel}_seed{a.seed}"
    log_path=os.path.join(a.out_dir,f"{slug}.log.json")
    log_data=[]
    start_step=0
    ckpt_files=sorted(glob.glob(os.path.join(a.out_dir,f"{slug}.step*.pt")),key=lambda p:int(p.rsplit('step',1)[1].rsplit('.pt',1)[0]))
    if ckpt_files:
        last=ckpt_files[-1]
        start_step=int(last.rsplit('step',1)[1].rsplit('.pt',1)[0])
        inv.load_state_dict(torch.load(last,map_location=dev))
        for _ in range(start_step):sched.step()
        if os.path.exists(log_path):
            try:log_data=json.load(open(log_path))
            except Exception:log_data=[]
        print(f"RESUME from {last} at step {start_step}")
    t0=time.time()
    BOS=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    PAD=tok.pad_token_id
    def encode_to_z(ids,mask):
        with torch.no_grad():
            cap=[None]
            def hk(m,i,o,c=cap):
                c[0]=(o[0] if isinstance(o,tuple) else o).detach()
            h=blk.register_forward_hook(hk)
            target(ids)
            h.remove()
            H=cap[0].float()
            mu,ls=qr.enc(H)
            z=mu+torch.randn_like(mu)*(0.5*ls).exp()+a.sigma_rel*torch.randn_like(mu)
        return z
    for step in range(start_step,a.steps):
        ids,mask,lens=next(it)
        ids=ids.to(dev)
        mask=mask.to(dev)
        T=ids.size(1)
        z=encode_to_z(ids,mask)
        mech=torch.full((ids.size(0),),a.mech_id,device=dev,dtype=torch.long)
        sig=torch.zeros(ids.size(0),device=dev,dtype=torch.long)
        bos=torch.full((ids.size(0),1),BOS,device=dev,dtype=torch.long)
        tgt_in=torch.cat([bos,ids[:,:-1]],dim=1)
        logits=inv(z,mech,sig,tgt_in)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),ids.reshape(-1),ignore_index=PAD,reduction="none").reshape(ids.size())
        loss=(loss*mask.float()).sum()/mask.float().sum().clamp_min(1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inv.parameters(),1.0)
        opt.step()
        sched.step()
        if step%a.log_every==0:
            log_data.append(dict(step=step,loss=loss.item(),lr=sched.get_last_lr()[0]))
            print(f"step {step} loss={loss.item():.4f}")
            with open(log_path,"w") as f:json.dump(log_data,f,indent=2)
        if step>0 and step%a.ckpt_every==0:
            torch.save(inv.state_dict(),os.path.join(a.out_dir,f"{slug}.step{step}.pt"))
    final=dict(slug=slug,target_model=a.target_model,target_layer=a.target_layer,r=a.r,sigma_rel=a.sigma_rel,steps=a.steps,elapsed_s=time.time()-t0,inv_params=n_params,log=log_data)
    with open(os.path.join(a.out_dir,f"{slug}.json"),"w") as f:json.dump(final,f,indent=2)
    torch.save(inv.state_dict(),os.path.join(a.out_dir,f"{slug}.final.pt"))
    print(f"saved {slug}")

if __name__=="__main__":
    main()

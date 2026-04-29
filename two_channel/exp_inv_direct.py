import argparse,json,os,time,math,glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset,DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from .sequence_inverter import SeqInv
from .adjacency_builder_v2 import get_layer_block

class WikiTextDS(IterableDataset):
    _cache=None
    @classmethod
    def get_pretok(cls,tok,seq_len,n_samples):
        if cls._cache is not None:return cls._cache
        from datasets import load_dataset
        ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
        out=[]
        for row in ds:
            txt=row["text"].strip()
            if len(txt)<80:continue
            ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
            if len(ids)>=seq_len:
                out.append(ids)
                if len(out)>=n_samples:break
        cls._cache=out
        return out
    def __init__(self,tok,seq_len,n_samples=200000,seed=0):
        self.tok=tok
        self.seq_len=seq_len
        self.seed=seed
        self.cache=self.get_pretok(tok,seq_len,n_samples)
    def __iter__(self):
        rng=torch.Generator().manual_seed(self.seed)
        N=len(self.cache)
        while True:
            i=int(torch.randint(0,N,(1,),generator=rng).item())
            ids=self.cache[i]
            s=int(torch.randint(0,len(ids)-self.seq_len+1,(1,),generator=rng).item())
            yield torch.tensor(ids[s:s+self.seq_len],dtype=torch.long)

def collate(batch):
    return torch.stack(batch,dim=0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target_model",required=True)
    ap.add_argument("--target_layer",type=int,required=True)
    ap.add_argument("--defense",choices=["clean","sigma_diag","isotropic"],default="sigma_diag")
    ap.add_argument("--sigma",type=float,default=5.0)
    ap.add_argument("--F_diag_path",default="")
    ap.add_argument("--seq_len",type=int,default=32)
    ap.add_argument("--max_T",type=int,default=64)
    ap.add_argument("--dm",type=int,default=512)
    ap.add_argument("--nhead",type=int,default=8)
    ap.add_argument("--enc_layers",type=int,default=6)
    ap.add_argument("--dec_layers",type=int,default=6)
    ap.add_argument("--ff",type=int,default=2048)
    ap.add_argument("--drop",type=float,default=0.1)
    ap.add_argument("--batch_size",type=int,default=64)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--steps",type=int,default=50000)
    ap.add_argument("--warmup",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--target_dtype",default="bfloat16")
    ap.add_argument("--out_dir",default="artifacts/inv_direct")
    ap.add_argument("--log_every",type=int,default=500)
    ap.add_argument("--ckpt_every",type=int,default=10000)
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
    print(f"target d={d}")
    F_diag=None
    if a.defense=="sigma_diag":
        if a.F_diag_path and os.path.exists(a.F_diag_path):
            F_diag=torch.load(a.F_diag_path,map_location=dev).float().clamp(min=1e-6)
            print(f"loaded F_diag from {a.F_diag_path}, shape={F_diag.shape}")
        else:
            print("computing F_diag on the fly")
            F_diag=torch.ones(d,device=dev)
    inv=SeqInv(r=d,vocab=tok.vocab_size,dm=a.dm,nhead=a.nhead,enc_layers=a.enc_layers,dec_layers=a.dec_layers,ff=a.ff,drop=a.drop,max_T=a.max_T).to(dev)
    n_params=sum(p.numel() for p in inv.parameters())
    print(f"inverter params={n_params/1e6:.1f}M")
    opt=torch.optim.AdamW(inv.parameters(),lr=a.lr,weight_decay=0.01)
    def lr_at(step):
        if step<a.warmup:return step/a.warmup
        prog=(step-a.warmup)/max(1,a.steps-a.warmup)
        return 0.5*(1+math.cos(math.pi*min(1.0,prog)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_at)
    ds=WikiTextDS(tok,a.seq_len,seed=a.seed)
    loader=DataLoader(ds,batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    it=iter(loader)
    slug=f"{a.target_model.replace('/','_')}_L{a.target_layer}_def-{a.defense}_sig{a.sigma}_seed{a.seed}"
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
        print(f"RESUME from step {start_step}")
    BOS=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    PAD=tok.pad_token_id
    def get_noisy_h(ids):
        cap=[None]
        def hk(m,i,o,c=cap):c[0]=(o[0] if isinstance(o,tuple) else o).detach()
        h=blk.register_forward_hook(hk)
        with torch.no_grad():
            target(ids)
        h.remove()
        H=cap[0].float()
        if a.defense=="clean":
            return H
        elif a.defense=="isotropic":
            return H+a.sigma*torch.randn_like(H)
        elif a.defense=="sigma_diag":
            std=a.sigma*F_diag.pow(-0.5)
            return H+torch.randn_like(H)*std[None,None,:]
    t0=time.time()
    for step in range(start_step,a.steps):
        ids=next(it).to(dev)
        z=get_noisy_h(ids)
        mech=torch.zeros(ids.size(0),device=dev,dtype=torch.long)
        sig=torch.zeros(ids.size(0),device=dev,dtype=torch.long)
        bos=torch.full((ids.size(0),1),BOS,device=dev,dtype=torch.long)
        tgt_in=torch.cat([bos,ids[:,:-1]],dim=1)
        logits=inv(z,mech,sig,tgt_in)
        loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),ids.reshape(-1),ignore_index=PAD if PAD is not None else -100)
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
    final=dict(slug=slug,target_model=a.target_model,target_layer=a.target_layer,defense=a.defense,sigma=a.sigma,steps=a.steps,inv_params=n_params,elapsed_s=time.time()-t0,log=log_data)
    with open(os.path.join(a.out_dir,f"{slug}.json"),"w") as f:json.dump(final,f,indent=2)
    torch.save(inv.state_dict(),os.path.join(a.out_dir,f"{slug}.final.pt"))
    print(f"saved {slug}")

if __name__=="__main__":
    main()

import argparse,json,os,time,math,glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset,DataLoader
from .split_memory_transformer import SMT,hutchinson_logit_v_jac

class GPTBaseline(nn.Module):
    def __init__(self,vocab,d=768,n_layers=12,n_heads=12,ff=3072,max_T=256):
        super().__init__()
        self.tok=nn.Embedding(vocab,d)
        self.pos=nn.Embedding(max_T,d)
        from .split_memory_transformer import Block
        self.layers=nn.ModuleList([Block(d,n_heads,ff) for _ in range(n_layers)])
        self.lnf=nn.LayerNorm(d)
        self.head=nn.Linear(d,vocab,bias=False)
        self.head.weight=self.tok.weight
    def forward(self,ids):
        B,T=ids.shape
        x=self.tok(ids)+self.pos(torch.arange(T,device=ids.device))[None]
        for L in self.layers:
            x=L(x)
        return self.head(self.lnf(x))

class WikiTextStream(IterableDataset):
    _cache=None
    @classmethod
    def get(cls,tok,seq_len,n_samples,corpus="wikitext"):
        if cls._cache is not None:return cls._cache
        from datasets import load_dataset
        if corpus=="wikitext":
            ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
        elif corpus=="tinystories":
            ds=load_dataset("roneneldan/TinyStories",split="train")
        else:
            raise ValueError(corpus)
        out=[]
        for row in ds:
            txt=row.get("text","").strip()
            if len(txt)<80:continue
            ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
            if len(ids)>=seq_len:
                out.append(ids)
                if len(out)>=n_samples:break
        cls._cache=out
        return out
    def __init__(self,tok,seq_len,n_samples=200000,seed=0,corpus="wikitext"):
        self.tok=tok
        self.seq_len=seq_len
        self.seed=seed
        self.cache=self.get(tok,seq_len,n_samples,corpus)
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
    ap.add_argument("--arch",choices=["smt","baseline"],default="smt")
    ap.add_argument("--corpus",choices=["wikitext","tinystories"],default="wikitext")
    ap.add_argument("--r",type=int,default=128)
    ap.add_argument("--m",type=int,default=640)
    ap.add_argument("--n_layers",type=int,default=12)
    ap.add_argument("--hr",type=int,default=4)
    ap.add_argument("--hm",type=int,default=4)
    ap.add_argument("--ff_r",type=int,default=512)
    ap.add_argument("--ff_m",type=int,default=1280)
    ap.add_argument("--max_T",type=int,default=256)
    ap.add_argument("--seq_len",type=int,default=256)
    ap.add_argument("--batch_size",type=int,default=32)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--warmup",type=int,default=1000)
    ap.add_argument("--lambda_jac",type=float,default=0.0)
    ap.add_argument("--probe_layers",type=str,default="4,6,8")
    ap.add_argument("--n_train_samples",type=int,default=100000)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--dtype",default="bfloat16")
    ap.add_argument("--out_dir",default="artifacts/smt")
    ap.add_argument("--log_every",type=int,default=200)
    ap.add_argument("--ckpt_every",type=int,default=5000)
    ap.add_argument("--tag",default="main")
    a=ap.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    torch.manual_seed(a.seed)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    dtype=getattr(torch,a.dtype)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained("openai-community/gpt2")
    if a.arch=="smt":
        model=SMT(vocab=tok.vocab_size,r=a.r,m=a.m,n_layers=a.n_layers,hr=a.hr,hm=a.hm,ff_r=a.ff_r,ff_m=a.ff_m,max_T=a.max_T).to(dev)
    else:
        d_total=a.r+a.m
        model=GPTBaseline(vocab=tok.vocab_size,d=d_total,n_layers=a.n_layers,n_heads=8,ff=4*d_total,max_T=a.max_T).to(dev)
    n_params=sum(p.numel() for p in model.parameters())
    print(f"arch={a.arch} params={n_params/1e6:.1f}M")
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.1,betas=(0.9,0.95))
    def lr_at(step):
        if step<a.warmup:return step/a.warmup
        prog=(step-a.warmup)/max(1,a.steps-a.warmup)
        return 0.5*(1+math.cos(math.pi*min(1.0,prog)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_at)
    ds=WikiTextStream(tok,a.seq_len,n_samples=a.n_train_samples,seed=a.seed,corpus=a.corpus)
    loader=DataLoader(ds,batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    it=iter(loader)
    slug=f"{a.arch}_{a.tag}_r{a.r}_m{a.m}_lj{a.lambda_jac}_seed{a.seed}"
    log_path=os.path.join(a.out_dir,f"{slug}.log.json")
    log_data=[]
    start_step=0
    ckpt_files=sorted(glob.glob(os.path.join(a.out_dir,f"{slug}.step*.pt")),key=lambda p:int(p.rsplit('step',1)[1].rsplit('.pt',1)[0]))
    if ckpt_files:
        last=ckpt_files[-1]
        start_step=int(last.rsplit('step',1)[1].rsplit('.pt',1)[0])
        sd=torch.load(last,map_location=dev)
        model.load_state_dict(sd)
        for _ in range(start_step):sched.step()
        if os.path.exists(log_path):
            try:log_data=json.load(open(log_path))
            except Exception:log_data=[]
        print(f"RESUME from step {start_step}")
    probe_layers=[int(x) for x in a.probe_layers.split(",")]
    t0=time.time()
    for step in range(start_step,a.steps):
        ids=next(it).to(dev)
        if a.arch=="smt" and a.lambda_jac>0:
            logits,u_l,v_l=model(ids,return_uv=True)
        else:
            logits=model(ids)
        L_lm=F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),ids[:,1:].reshape(-1))
        L=L_lm
        if a.arch=="smt" and a.lambda_jac>0:
            xi=torch.randn_like(logits)
            s=(logits*xi).sum()
            grads=torch.autograd.grad(s,[v_l[i] for i in probe_layers],retain_graph=True,create_graph=True,allow_unused=True)
            jpen=logits.new_zeros(())
            for g in grads:
                if g is not None:jpen=jpen+g.pow(2).sum()/g.numel()
            L=L+a.lambda_jac*jpen
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        sched.step()
        if step%a.log_every==0:
            entry=dict(step=step,L=L.item(),L_lm=L_lm.item(),lr=sched.get_last_lr()[0])
            if a.arch=="smt" and a.lambda_jac>0:
                entry["jpen"]=jpen.item() if hasattr(jpen,"item") else 0.0
            log_data.append(entry)
            print(f"step {step} L={L.item():.4f} L_lm={L_lm.item():.4f}")
            with open(log_path,"w") as f:json.dump(log_data,f,indent=2)
        if step>0 and step%a.ckpt_every==0:
            torch.save(model.state_dict(),os.path.join(a.out_dir,f"{slug}.step{step}.pt"))
    final=dict(slug=slug,arch=a.arch,r=a.r,m=a.m,n_layers=a.n_layers,n_params=n_params,steps=a.steps,lambda_jac=a.lambda_jac,elapsed_s=time.time()-t0,log=log_data)
    with open(os.path.join(a.out_dir,f"{slug}.json"),"w") as f:json.dump(final,f,indent=2)
    torch.save(model.state_dict(),os.path.join(a.out_dir,f"{slug}.final.pt"))
    print(f"saved {slug}")

if __name__=="__main__":
    main()

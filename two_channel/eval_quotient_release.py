import argparse,json,os,time,glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from .quotient_release import QuotientRelease
from .adjacency_builder_v2 import get_layer_block

def make_ds(tok,n,sl,seed=42):
    from datasets import load_dataset
    ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    torch.manual_seed(seed)
    out=[]
    for row in ds:
        txt=row["text"].strip()
        if len(txt)<80:continue
        ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=sl:
            s=torch.randint(0,len(ids)-sl+1,(1,)).item()
            out.append(torch.tensor(ids[s:s+sl],dtype=torch.long))
            if len(out)>=n:break
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--layer",type=int,required=True)
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--r",type=int,required=True)
    ap.add_argument("--n_bank",type=int,default=50000)
    ap.add_argument("--n_query",type=int,default=2000)
    ap.add_argument("--seq_len",type=int,default=64)
    ap.add_argument("--sigma_rel",type=float,default=0.2)
    ap.add_argument("--ctx_for_eval",type=int,default=32)
    ap.add_argument("--dtype",default="float32")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    dtype=getattr(torch,a.dtype)
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype).to(dev).eval()
    blk=get_layer_block(model,a.layer)
    d=model.config.hidden_size
    qr=QuotientRelease(d,a.r).to(dev).to(torch.float32)
    qr.load_state_dict(torch.load(a.ckpt,map_location=dev))
    qr.eval()
    print("dataset")
    ds=make_ds(tok,a.n_bank+a.n_query,a.ctx_for_eval)
    bank_pref=ds[:a.n_bank-a.n_query] if a.n_bank>a.n_query else ds[:a.n_bank]
    query_pref=ds[-a.n_query:]
    def hidden_at_layer(ids):
        cap=[None]
        def hk(m,i,o,c=cap):
            c[0]=(o[0] if isinstance(o,tuple) else o).detach()
        h=blk.register_forward_hook(hk)
        with torch.no_grad():model(ids)
        h.remove()
        return cap[0].float()
    print("embedding bank")
    H_query=[]
    for p in tqdm(query_pref,desc="query"):
        H_query.append(hidden_at_layer(p.unsqueeze(0).to(dev))[:,-1,:].cpu().squeeze(0))
    H_query=torch.stack(H_query)
    H_bank_dist=[]
    for p in tqdm(bank_pref,desc="bank"):
        H_bank_dist.append(hidden_at_layer(p.unsqueeze(0).to(dev))[:,-1,:].cpu().squeeze(0))
    H_bank_dist=torch.stack(H_bank_dist) if H_bank_dist else torch.zeros(0,d)
    print("encoding to z")
    with torch.no_grad():
        Z_query=[]
        Hq=H_query.unsqueeze(1).to(dev)
        for i in range(0,Hq.size(0),128):
            mu,ls=qr.enc(Hq[i:i+128])
            Z_query.append(mu.squeeze(1).cpu())
        Z_query=torch.cat(Z_query,dim=0)
        Z_bank_dist=[]
        Hb=H_bank_dist.unsqueeze(1).to(dev) if H_bank_dist.numel()>0 else None
        if Hb is not None:
            for i in range(0,Hb.size(0),128):
                mu,ls=qr.enc(Hb[i:i+128])
                Z_bank_dist.append(mu.squeeze(1).cpu())
            Z_bank_dist=torch.cat(Z_bank_dist,dim=0)
        else:
            Z_bank_dist=torch.zeros(0,a.r)
    Z_bank=torch.cat([Z_query,Z_bank_dist],dim=0)
    gt_idx=torch.arange(a.n_query)
    print("retrieval")
    Z_bank_dev=Z_bank.to(dev)
    correct_l2=0
    correct_mah=0
    torch.manual_seed(0)
    noise=torch.randn(Z_query.shape)*a.sigma_rel
    Z_noisy=(Z_query+noise).to(dev)
    cb_l2=(Z_bank_dev*Z_bank_dev).sum(-1)
    bsz=64
    N=Z_noisy.size(0)
    for i in range(0,N,bsz):
        q=Z_noisy[i:i+bsz]
        cross=q@Z_bank_dev.T
        d_l2=(q*q).sum(-1,keepdim=True)+cb_l2.unsqueeze(0)-2*cross
        idx=d_l2.argmin(dim=-1).cpu()
        correct_l2+=(idx==gt_idx[i:i+bsz]).sum().item()
    inv=1.0/(a.sigma_rel**2+1e-8)
    cb_mah=cb_l2*inv
    for i in range(0,N,bsz):
        q=Z_noisy[i:i+bsz]
        cross=q@Z_bank_dev.T
        d_mah=((q*q).sum(-1,keepdim=True)+cb_l2.unsqueeze(0)-2*cross)*inv
        idx=d_mah.argmin(dim=-1).cpu()
        correct_mah+=(idx==gt_idx[i:i+bsz]).sum().item()
    print("utility")
    util_kls=[]
    t1_match=0
    n_util=200
    for p in tqdm(query_pref[:n_util],desc="util"):
        ids=p.unsqueeze(0).to(dev)
        h_clean=hidden_at_layer(ids)
        with torch.no_grad():
            mu,ls=qr.enc(h_clean)
            z=mu+torch.randn_like(mu)*(0.5*ls).exp()+a.sigma_rel*torch.randn_like(mu)
            h_hat=qr.dec(z)
        def inj(m,i,o,hn=h_hat):
            oo=o[0] if isinstance(o,tuple) else o
            oo=hn.to(oo.dtype)
            if isinstance(o,tuple):return(oo,)+o[1:]
            return oo
        h_=blk.register_forward_hook(inj)
        with torch.no_grad():
            o_hat=model(ids)
        h_.remove()
        with torch.no_grad():
            o_clean=model(ids)
        lp_c=F.log_softmax(o_clean.logits[0,-1].float(),dim=-1)
        lp_h=F.log_softmax(o_hat.logits[0,-1].float(),dim=-1)
        kl=(lp_c.exp()*(lp_c-lp_h)).sum().item()
        util_kls.append(kl)
        if lp_c.argmax()==lp_h.argmax():t1_match+=1
    res=dict(model=a.model,layer=a.layer,r=a.r,sigma_rel=a.sigma_rel,n_bank=a.n_bank,n_query=a.n_query,
             attack_top1_l2=correct_l2/N,attack_top1_mah=correct_mah/N,
             mean_kl=sum(util_kls)/len(util_kls),t1_agreement=t1_match/n_util,
             moderate_both=(t1_match/n_util>=0.5 and max(correct_l2,correct_mah)/N<=0.5))
    os.makedirs(os.path.dirname(a.out)or".",exist_ok=True)
    with open(a.out,"w") as f:json.dump(res,f,indent=2)
    print(json.dumps(res,indent=2))

if __name__=="__main__":
    main()

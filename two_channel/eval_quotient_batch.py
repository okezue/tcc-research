import argparse,json,os,glob,time
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

def hidden_at_layer(model,blk,ids):
    cap=[None]
    def hk(m,i,o,c=cap):c[0]=(o[0] if isinstance(o,tuple) else o).detach()
    h=blk.register_forward_hook(hk)
    with torch.no_grad():model(ids)
    h.remove()
    return cap[0][:,-1,:].float()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="openai-community/gpt2")
    ap.add_argument("--layer",type=int,default=6)
    ap.add_argument("--ckpt_dir",default="artifacts/quotient_release")
    ap.add_argument("--n_bank",type=int,default=50000)
    ap.add_argument("--n_query",type=int,default=2000)
    ap.add_argument("--ctx",type=int,default=32)
    ap.add_argument("--out",default="artifacts/quotient_release/eval_summary.json")
    ap.add_argument("--dtype",default="float32")
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    dtype=getattr(torch,a.dtype)
    print(f"loading {a.model}")
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype).to(dev).eval()
    blk=get_layer_block(model,a.layer)
    d=model.config.hidden_size
    print(f"d={d}")
    print(f"building dataset")
    ds=make_ds(tok,a.n_bank+a.n_query,a.ctx)
    bank_pref=ds[:a.n_bank-a.n_query] if a.n_bank>a.n_query else ds[:a.n_bank]
    query_pref=ds[-a.n_query:]
    print(f"embedding bank ({len(bank_pref)} prefixes)")
    H_bank_dist=[]
    for p in tqdm(bank_pref,desc="bank"):
        H_bank_dist.append(hidden_at_layer(model,blk,p.unsqueeze(0).to(dev)).cpu().squeeze(0))
    H_bank_dist=torch.stack(H_bank_dist) if H_bank_dist else torch.zeros(0,d)
    print(f"embedding queries ({len(query_pref)})")
    H_query=[]
    Q_logits=[]
    for p in tqdm(query_pref,desc="query"):
        ids=p.unsqueeze(0).to(dev)
        cap=[None]
        def hk(m,i,o,c=cap):c[0]=(o[0] if isinstance(o,tuple) else o).detach()
        h=blk.register_forward_hook(hk)
        with torch.no_grad():out=model(ids)
        h.remove()
        H_query.append(cap[0][0,-1,:].float().cpu())
        Q_logits.append(out.logits[0,-1,:].float().cpu())
    H_query=torch.stack(H_query)
    Q_logits=torch.stack(Q_logits)
    H_query_dev=H_query.to(dev)
    H_bank_dist_dev=H_bank_dist.to(dev)
    final_ckpts=sorted(glob.glob(os.path.join(a.ckpt_dir,"*.final.pt")))
    print(f"found {len(final_ckpts)} ckpts")
    rows=[]
    for ck in final_ckpts:
        slug=os.path.basename(ck).replace(".final.pt","")
        info_json=os.path.join(a.ckpt_dir,f"{slug}.json")
        if not os.path.exists(info_json):continue
        d_info=json.load(open(info_json))
        r=d_info["r"]
        sigma_rel=d_info.get("sigma_rel",0.2)
        qr=QuotientRelease(d,r).to(dev).to(torch.float32).eval()
        qr.load_state_dict(torch.load(ck,map_location=dev))
        with torch.no_grad():
            mu_b,_=qr.enc(H_bank_dist_dev.unsqueeze(1))
            Z_bank_dist=mu_b.squeeze(1)
            mu_q,ls_q=qr.enc(H_query_dev.unsqueeze(1))
            Z_query=mu_q.squeeze(1)
        Z_bank=torch.cat([Z_query,Z_bank_dist],dim=0)
        gt_idx=torch.arange(a.n_query,device=dev)
        torch.manual_seed(0)
        if sigma_rel>0:
            noise=torch.randn(Z_query.shape,device=dev)*sigma_rel
            Z_noisy=Z_query+noise
        else:
            Z_noisy=Z_query
        cb_l2=(Z_bank*Z_bank).sum(-1)
        bsz=64
        N=Z_noisy.size(0)
        correct_l2=0
        for i in range(0,N,bsz):
            q=Z_noisy[i:i+bsz]
            cross=q@Z_bank.T
            d_l2=(q*q).sum(-1,keepdim=True)+cb_l2.unsqueeze(0)-2*cross
            idx=d_l2.argmin(dim=-1)
            correct_l2+=(idx==gt_idx[i:i+bsz]).sum().item()
        n_util=200
        ids_util=query_pref[:n_util]
        kls=[];t1=0
        for j in range(n_util):
            ids=ids_util[j].unsqueeze(0).to(dev)
            cap=[None]
            def hk(m,i,o,c=cap):c[0]=(o[0] if isinstance(o,tuple) else o).detach()
            h_=blk.register_forward_hook(hk)
            with torch.no_grad():o_clean=model(ids)
            h_.remove()
            H_orig=cap[0].float()
            with torch.no_grad():
                mu_seq,ls_seq=qr.enc(H_orig)
                z_seq=mu_seq+torch.randn_like(mu_seq)*(0.5*ls_seq).exp()
                if sigma_rel>0:
                    z_seq=z_seq+sigma_rel*torch.randn_like(z_seq)
                h_hat_seq=qr.dec(z_seq)
            def inj(m,i,o,hh=h_hat_seq):
                oo=o[0] if isinstance(o,tuple) else o
                oo=hh.to(oo.dtype)
                if isinstance(o,tuple):return(oo,)+o[1:]
                return oo
            h2=blk.register_forward_hook(inj)
            with torch.no_grad():o_hat=model(ids)
            h2.remove()
            lp_c=F.log_softmax(o_clean.logits[0,-1].float(),dim=-1)
            lp_h=F.log_softmax(o_hat.logits[0,-1].float(),dim=-1)
            kl=(lp_c.exp()*(lp_c-lp_h)).sum().item()
            kls.append(kl)
            if lp_c.argmax()==lp_h.argmax():t1+=1
        rec=dict(slug=slug,r=r,beta=d_info.get("beta"),gamma=d_info.get("gamma"),sigma_rel=sigma_rel,
                 attack_top1=correct_l2/N,
                 mean_kl=sum(kls)/len(kls),
                 t1_agreement=t1/n_util,
                 moderate_both=(t1/n_util>=0.5 and correct_l2/N<=0.5))
        rows.append(rec)
        print(f"{slug[:60]}: t1={rec['t1_agreement']:.3f} attack={rec['attack_top1']:.3f} kl={rec['mean_kl']:.3f}")
    out=dict(model=a.model,layer=a.layer,n_bank=a.n_bank,n_query=a.n_query,n_util=200,results=rows)
    with open(a.out,"w") as f:json.dump(out,f,indent=2)
    print(f"\nwrote {a.out}")
    n_mod=sum(1 for r in rows if r["moderate_both"])
    print(f"moderate-both cells: {n_mod}/{len(rows)}")

if __name__=="__main__":
    main()

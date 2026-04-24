#!/usr/bin/env python3
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import json,time,gc,argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/multi_horizon")
OUT.mkdir(parents=True,exist_ok=True)

def make_ds(tok,n=3000,sl=129,seed=42):
    from datasets import load_dataset
    ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    torch.manual_seed(seed)
    out=[]
    for row in ds:
        txt=row["text"].strip()
        if len(txt)<200: continue
        ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=sl:
            s=torch.randint(0,len(ids)-sl+1,(1,)).item()
            out.append(torch.tensor(ids[s:s+sl],dtype=torch.long))
            if len(out)>=n: break
    return out

def get_layer_block(model,layer):
    if hasattr(model,'transformer') and hasattr(model.transformer,'h'):
        return model.transformer.h[layer],layer,len(model.transformer.h)
    raise ValueError("cannot find layers")

def compute_grad_cov_H(model,ds,layer,H,dev,n_cal=2000,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    d=model.config.hidden_size
    G=torch.zeros(d,d,dtype=torch.float64)
    cnt=0
    for seq in tqdm(ds[:n_cal],desc=f"grad cov H={H}"):
        if len(seq)<ctx+H: continue
        x=seq[:ctx].unsqueeze(0).to(dev)
        ys=seq[ctx:ctx+H].to(dev)
        cap=[None]
        def hk(m,i,o,c=cap):
            c[0]=o[0] if isinstance(o,tuple) else o
        h=blk.register_forward_hook(hk)
        with torch.no_grad(): model(x)
        h.remove()
        h_last=cap[0][0,-1,:].detach().float()
        h_last_grad=h_last.clone().requires_grad_(True)
        cur_h=h_last_grad.unsqueeze(0)
        total_loss=0
        layers_after=list(model.transformer.h[layer+1:]) if hasattr(model,'transformer') else []
        x_seq=seq[:ctx+H-1].unsqueeze(0).to(dev)
        def inject(m,i,o):
            oo=o[0] if isinstance(o,tuple) else o
            oo=oo.clone()
            oo[0,ctx-1,:]=h_last_grad.to(oo.dtype)
            if isinstance(o,tuple): return (oo,)+o[1:]
            return oo
        hh=blk.register_forward_hook(inject)
        out=model(x_seq)
        hh.remove()
        all_logits=out.logits[0,ctx-1:ctx+H-1,:]
        total_loss=F.cross_entropy(all_logits,ys)
        g=torch.autograd.grad(total_loss,h_last_grad)[0].detach().float().cpu()
        G+=torch.outer(g.to(torch.float64),g.to(torch.float64))
        cnt+=1
    G/=cnt
    evals,evecs=torch.linalg.eigh(G.float())
    idx=evals.argsort(descending=True)
    return evecs[:,idx],evals[idx]

def embed_bank(model,ds,layer,dev,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    hs=[]
    cap=[None]
    def hk(m,i,o,c=cap):
        c[0]=o[0].detach() if isinstance(o,tuple) else o.detach()
    h=blk.register_forward_hook(hk)
    for seq in tqdm(ds,desc="embed",leave=False):
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad(): model(x)
        hs.append(cap[0][0,-1,:].float().cpu())
    h.remove()
    return torch.stack(hs)

def compute_margins(H,U):
    d=H.shape[1]
    H_B=H@U@U.T
    H_I=H-H_B
    D_full=torch.cdist(H,H);D_full.fill_diagonal_(float('inf'))
    D_B=torch.cdist(H_B,H_B);D_B.fill_diagonal_(float('inf'))
    D_I=torch.cdist(H_I,H_I);D_I.fill_diagonal_(float('inf'))
    fm=D_full.min(dim=1).values
    bm=D_B.min(dim=1).values
    im=D_I.min(dim=1).values
    return {
        "full_margin":float(fm.median()),
        "beh_margin":float(bm.median()),
        "id_margin":float(im.median()),
        "beh_frac":float((bm.median()/fm.median()).item()),
        "id_frac":float((im.median()/fm.median()).item()),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    mname=args.model
    dev=args.device
    t0=time.time()

    print("="*60+f"\nMulti-Horizon Predictive Quotient: {mname}\n"+"="*60)

    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(mname,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    model=AutoModelForCausalLM.from_pretrained(mname,torch_dtype=torch.float32,output_hidden_states=True,trust_remote_code=True)
    model.eval().to(dev)
    for pp in model.parameters(): pp.requires_grad_(False)

    d=model.config.hidden_size
    layer=args.layer
    k=args.k
    ctx=32

    H_horizons=[1,4,16,32]
    ds=make_ds(tok,n=3000,sl=ctx+max(H_horizons)+1)
    print(f"Dataset: {len(ds)} prefixes of length {ctx+max(H_horizons)+1}")

    print("\n--- Embedding bank ---")
    H_bank=embed_bank(model,ds[:1000],layer,dev,ctx=ctx)

    results=[]
    U_H={}
    for H in H_horizons:
        print(f"\n--- H={H} gradient covariance ---")
        evecs,evals=compute_grad_cov_H(model,ds,layer,H,dev,n_cal=2000,ctx=ctx)
        U_H[H]=evecs[:,:k]
        cs=evals.cumsum(0)/evals.sum()
        r95=0
        for i,c in enumerate(cs):
            if c>=0.95: r95=i+1; break
        energy_k=float(cs[k-1].item())
        margins=compute_margins(H_bank,U_H[H])
        results.append({"H":H,"r95":r95,"energy_top_k":energy_k,**margins})
        print(f"  Energy@k={k}: {energy_k:.3f}, r95={r95}")
        print(f"  beh_frac={margins['beh_frac']:.3f} id_frac={margins['id_frac']:.3f}")

    print(f"\n--- Subspace overlap between horizons ---")
    overlap_matrix={}
    for H1 in H_horizons:
        for H2 in H_horizons:
            if H1>=H2: continue
            U1=U_H[H1];U2=U_H[H2]
            M=U1.T@U2
            svals=torch.linalg.svdvals(M)
            avg_cos=float(svals.mean())
            overlap_matrix[f"{H1}_{H2}"]=avg_cos
            print(f"  H={H1} vs H={H2}: mean(σ)={avg_cos:.3f}")

    print(f"\n--- Within-class vs between-class variance (H=4 behavior classes) ---")
    H_check=4
    if H_check in U_H:
        if len(ds)<500+max(H_horizons): ds_extra=ds
        else: ds_extra=ds[:1500]
        tau=0.5
        bank_ds=ds_extra[:1000]
        H_embed=H_bank
        b_sigs=[]
        for seq in tqdm(bank_ds[:500],desc=f"H={H_check} sigs",leave=False):
            if len(seq)<ctx+H_check: continue
            x=seq[:ctx].unsqueeze(0).to(dev)
            with torch.no_grad():
                full_x=seq[:ctx+H_check-1].unsqueeze(0).to(dev)
                out=model(full_x)
                logits=out.logits[0,ctx-1:ctx+H_check-1,:]
                b_sigs.append(F.log_softmax(logits,dim=-1).cpu())
        n_sigs=len(b_sigs)
        D_kl=torch.zeros(n_sigs,n_sigs)
        for i in range(n_sigs):
            for j in range(n_sigs):
                if i==j: continue
                for t in range(H_check):
                    D_kl[i,j]+=F.kl_div(b_sigs[j][t],b_sigs[i][t].exp(),reduction='sum',log_target=False).item()
        classes=[]
        visited=set()
        for i in range(n_sigs):
            if i in visited: continue
            cls=[i]
            for j in range(i+1,n_sigs):
                if j not in visited and D_kl[i,j]<tau:
                    cls.append(j)
                    visited.add(j)
            visited.add(i)
            if len(cls)>=2: classes.append(cls)
        print(f"  Found {len(classes)} classes of size >=2")
        U_H4=U_H[H_check].to(torch.float32)
        P_B=U_H4@U_H4.T
        P_I=torch.eye(d)-P_B
        H_use=H_embed[:n_sigs]
        W_total=torch.zeros(d,d,dtype=torch.float64)
        B_between=torch.zeros(d,d,dtype=torch.float64)
        n_W=0
        for cls in classes:
            if len(cls)<2: continue
            hs_cls=H_use[cls]
            mu=hs_cls.mean(0)
            for h in hs_cls:
                diff=(h-mu).double()
                W_total+=torch.outer(diff,diff)
                n_W+=1
        if n_W>0: W_total/=n_W
        mu_global=H_use.mean(0)
        n_B=0
        for cls in classes:
            if len(cls)<2: continue
            mu_cls=H_use[cls].mean(0)
            diff=(mu_cls-mu_global).double()
            B_between+=torch.outer(diff,diff)
            n_B+=1
        if n_B>0: B_between/=n_B
        W=W_total.float()
        B_=B_between.float()
        within_I_frac=float(torch.trace(P_I@W)/max(torch.trace(W).item(),1e-8))
        within_B_frac=float(torch.trace(P_B@W)/max(torch.trace(W).item(),1e-8))
        between_B_frac=float(torch.trace(P_B@B_)/max(torch.trace(B_).item(),1e-8))
        between_I_frac=float(torch.trace(P_I@B_)/max(torch.trace(B_).item(),1e-8))
        print(f"  Within-class variance in P_I: {within_I_frac:.3f} (vs P_B: {within_B_frac:.3f})")
        print(f"  Between-class variance in P_B: {between_B_frac:.3f} (vs P_I: {between_I_frac:.3f})")
        variance_results={
            "n_classes":len(classes),
            "within_I_frac":within_I_frac,
            "within_B_frac":within_B_frac,
            "between_B_frac":between_B_frac,
            "between_I_frac":between_I_frac,
        }
    else:
        variance_results={}

    output={
        "model":mname,"layer":layer,"k":k,"d":d,"ctx":ctx,
        "horizons":results,
        "subspace_overlap":overlap_matrix,
        "variance_partition":variance_results,
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s":time.time()-t0,
    }
    slug=mname.replace("/","_")
    with open(OUT/f"multi_horizon_{slug}.json","w") as f:
        json.dump(output,f,indent=2)
    print(f"\nResults saved to {OUT}")
    print(f"Total time: {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()

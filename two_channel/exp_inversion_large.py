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
OUT=Path("artifacts/inversion_large")
OUT.mkdir(parents=True,exist_ok=True)

def make_ds(tok,n=5000,sl=33,seed=42):
    from datasets import load_dataset
    ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    torch.manual_seed(seed)
    out=[]
    for row in ds:
        txt=row["text"].strip()
        if len(txt)<80: continue
        ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=sl:
            s=torch.randint(0,len(ids)-sl+1,(1,)).item()
            out.append(torch.tensor(ids[s:s+sl],dtype=torch.long))
            if len(out)>=n: break
    return out

def get_layer_block(model,layer):
    if hasattr(model,'transformer') and hasattr(model.transformer,'h'):
        return model.transformer.h[layer],layer,len(model.transformer.h)
    if hasattr(model,'model') and hasattr(model.model,'layers'):
        return model.model.layers[layer],layer,len(model.model.layers)
    raise ValueError("Cannot find layers")

def embed_all(model,ds,layer,dev,U_B=None,mode="full",ctx=32):
    blk,_,_=get_layer_block(model,layer)
    hs=[]
    cap=[None]
    def hk(m,i,o,c=cap):
        oo=o[0] if isinstance(o,tuple) else o
        c[0]=oo.detach()
    handle=blk.register_forward_hook(hk)
    for seq in tqdm(ds,desc=f"embed {mode}",leave=False):
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad(): model(x)
        h=cap[0][0,-1,:].float()
        if mode=="behavior" and U_B is not None:
            h=h@U_B@U_B.T
        elif mode=="identity" and U_B is not None:
            h=h-h@U_B@U_B.T
        hs.append(h.cpu())
    handle.remove()
    return torch.stack(hs)

def retrieval_eval(H_query,H_bank,query_idx):
    D=torch.cdist(H_query,H_bank)
    n_q=H_query.shape[0]
    n_bank=H_bank.shape[0]
    ranks=[]
    for i in range(n_q):
        qi=query_idx[i]
        row=D[i]
        rank=(row<row[qi]).sum().item()+1
        ranks.append(rank)
    top1=sum(1 for r in ranks if r==1)/n_q
    mrr=sum(1.0/r for r in ranks)/n_q
    r10=sum(1 for r in ranks if r<=10)/n_q
    r50=sum(1 for r in ranks if r<=50)/n_q
    return {"top1":top1,"mrr":mrr,"recall10":r10,"recall50":r50,
            "median_rank":float(np.median(ranks)),"mean_rank":float(np.mean(ranks)),
            "p90_rank":float(np.percentile(ranks,90))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=None)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--n-bank",type=int,default=5000)
    p.add_argument("--n-query",type=int,default=500)
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    mname=args.model
    dev=args.device
    t0=time.time()

    print("="*60+f"\nLarge-Distractor Inversion: {mname}\n"+"="*60)

    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(mname,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    is_gpt2="gpt2" in mname.lower()
    dtype=torch.float32 if is_gpt2 else torch.float16
    model=AutoModelForCausalLM.from_pretrained(mname,torch_dtype=dtype,output_hidden_states=True,trust_remote_code=True)
    model.eval().to(dev)
    for pp in model.parameters(): pp.requires_grad_(False)

    d=model.config.hidden_size
    nl=model.config.num_hidden_layers
    layer=args.layer if args.layer else nl//2
    k=args.k
    ctx=32
    n_bank=args.n_bank
    n_query=args.n_query
    print(f"d={d} layers={nl} layer={layer} k={k} ctx={ctx} bank={n_bank} query={n_query}")

    ds=make_ds(tok,n=n_bank+n_query+500,sl=ctx+1)
    bank_ds=ds[:n_bank]
    query_ds=ds[n_bank:n_bank+n_query]

    sp=Path(f"artifacts/subspace/{mname.replace('/','_')}/layer_{layer}")
    if not sp.exists():
        for candidate in [6,11,layer]:
            sp2=Path(f"artifacts/subspace/{mname.replace('/','_')}/layer_{candidate}")
            if sp2.exists(): sp=sp2; break
    loaded=False
    if sp.exists():
        try:
            evecs=torch.load(sp/"grad_evecs.pt",weights_only=True)
            U_B=evecs[:,:k].to(torch.float32).to(dev)
            print(f"Loaded subspace from {sp}")
            loaded=True
        except: pass
    if not loaded:
        print("Computing subspace on the fly...")
        try:
            from two_channel.exp_scaling_points import compute_grad_cov
            evecs,evals=compute_grad_cov(model,tok,bank_ds[:2000],layer,dev,n_cal=2000,ctx=ctx)
        except ImportError:
            from two_channel.compute_subspace import get_layer_block as glb
            blk_,_,_=glb(model,layer)
            dd=model.config.hidden_size
            G=torch.zeros(dd,dd,dtype=torch.float64)
            cnt=0
            for seq in tqdm(bank_ds[:2000],desc="grad cov"):
                x=seq[:ctx].unsqueeze(0).to(dev)
                y=seq[ctx].to(dev) if len(seq)>ctx else seq[-1].to(dev)
                cap=[None]
                def hk_(m,i,o,c=cap):
                    c[0]=(o[0] if isinstance(o,tuple) else o)
                hh=blk_.register_forward_hook(hk_)
                model(x); hh.remove()
                hl=cap[0][0,-1,:].detach().float().requires_grad_(True)
                if hasattr(model,'transformer'):
                    logits=model.lm_head(model.transformer.ln_f(hl))
                else:
                    logits=model.lm_head(model.model.norm(hl)) if hasattr(model.model,'norm') else model.lm_head(hl)
                loss=F.cross_entropy(logits.unsqueeze(0),y.unsqueeze(0))
                g=torch.autograd.grad(loss,hl)[0].detach().float().cpu()
                G+=torch.outer(g.to(torch.float64),g.to(torch.float64))
                cnt+=1
            G/=cnt
            evals_,evecs=torch.linalg.eigh(G.float())
            idx=evals_.argsort(descending=True)
            evecs=evecs[:,idx]
        U_B=evecs[:,:k].to(torch.float32).to(dev)

    print(f"\n--- Embedding bank ({n_bank}) ---")
    H_bank_full=embed_all(model,bank_ds,layer,dev,ctx=ctx)
    H_bank_beh=embed_all(model,bank_ds,layer,dev,U_B,mode="behavior",ctx=ctx)
    H_bank_id=embed_all(model,bank_ds,layer,dev,U_B,mode="identity",ctx=ctx)

    print(f"\n--- Embedding queries ({n_query}) ---")
    H_q_full=embed_all(model,query_ds,layer,dev,ctx=ctx)
    H_q_beh=embed_all(model,query_ds,layer,dev,U_B,mode="behavior",ctx=ctx)
    H_q_id=embed_all(model,query_ds,layer,dev,U_B,mode="identity",ctx=ctx)

    H_bank_full_ext=torch.cat([H_bank_full,H_q_full],dim=0)
    H_bank_beh_ext=torch.cat([H_bank_beh,H_q_beh],dim=0)
    H_bank_id_ext=torch.cat([H_bank_id,H_q_id],dim=0)
    query_idx=list(range(n_bank,n_bank+n_query))

    print(f"\n--- Retrieval evaluation (bank size={n_bank+n_query}) ---")
    results={}
    for mode,Hq,Hb in [("full",H_q_full,H_bank_full_ext),
                         ("behavior",H_q_beh,H_bank_beh_ext),
                         ("identity",H_q_id,H_bank_id_ext)]:
        r=retrieval_eval(Hq,Hb,query_idx)
        results[mode]=r
        print(f"  {mode:>10}: top1={r['top1']:.3f} mrr={r['mrr']:.3f} r@10={r['recall10']:.3f} r@50={r['recall50']:.3f} med_rank={r['median_rank']:.0f}")

    print(f"\n--- Noisy retrieval ---")
    noisy_results=[]
    for sigma in [0.1,0.5,1.0,2.0,5.0]:
        for mode in ["full","behavior","identity"]:
            if mode=="full":
                Hq=H_q_full+torch.randn_like(H_q_full)*sigma
                Hb=H_bank_full_ext
            elif mode=="behavior":
                Hq=H_q_beh+torch.randn_like(H_q_beh)*sigma
                Hb=H_bank_beh_ext
            else:
                Hq=H_q_id+torch.randn_like(H_q_id)*sigma
                Hb=H_bank_id_ext
            r=retrieval_eval(Hq,Hb,query_idx)
            r["sigma"]=sigma; r["mode"]=mode
            noisy_results.append(r)
            if mode in ["behavior","identity"]:
                print(f"  σ={sigma:.1f} {mode:>10}: top1={r['top1']:.3f} mrr={r['mrr']:.3f} r@10={r['recall10']:.3f} med_rank={r['median_rank']:.0f}")

    output={
        "model":mname,"layer":layer,"k":k,"d":d,"ctx":ctx,
        "n_bank":n_bank,"n_query":n_query,
        "clean_retrieval":results,
        "noisy_retrieval":noisy_results,
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s":time.time()-t0,
    }
    slug=mname.replace("/","_")
    with open(OUT/f"inversion_large_{slug}.json","w") as f:
        json.dump(output,f,indent=2)

    print(f"\nResults saved to {OUT}")
    print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""Test the fixed-projector isotropy theorem empirically.

For each model, estimate the normalized-margin covariance

    S_delta = E[ d_hat d_hat^T ],    d_hat = (h_x - h_x') / ||h_x - h_x'||,

from N random pairs, compute the isotropy error

    eps_iso = || d * S_delta - I ||_op,

and show that for any fixed rank-k projector P

    | tr(P S_delta) - k/d | <= (k/d) eps_iso.

This turns the current 'sqrt(k/d) fit' into a sufficient-condition test.
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch,json,time,argparse
from pathlib import Path
from two_channel.exp_optimal_defense import make_ds,embed_bank,get_layer_block
from transformers import AutoModelForCausalLM,AutoTokenizer

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/isotropy")
OUT.mkdir(parents=True,exist_ok=True)

def sample_margin_cov(H,n_pairs,seed=0):
    n,d=H.shape
    g=torch.Generator().manual_seed(seed)
    S=torch.zeros(d,d,dtype=torch.float64)
    for _ in range(n_pairs):
        i,j=torch.randint(0,n,(2,),generator=g).tolist()
        if i==j: continue
        delta=(H[i]-H[j]).double()
        nrm=delta.norm()
        if nrm<1e-8: continue
        dh=delta/nrm
        S+=torch.outer(dh,dh)
    return (S/n_pairs).float()

def random_projector(d,k,seed=0):
    g=torch.Generator().manual_seed(seed)
    A=torch.randn(d,k,generator=g)
    Q,_=torch.linalg.qr(A)
    return Q@Q.T

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--n",type=int,default=2000)
    p.add_argument("--n_pairs",type=int,default=20000)
    p.add_argument("--ks",default="32,64,128,256")
    p.add_argument("--ctx",type=int,default=32)
    args=p.parse_args()
    ks=[int(x) for x in args.ks.split(",")]
    t0=time.time()
    print(f"[isotropy] {args.model} L{args.layer} N={args.n} pairs={args.n_pairs}")
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    is_gpt2="gpt2" in args.model.lower()
    dtype=torch.float32 if is_gpt2 else torch.float16
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype,output_hidden_states=True,trust_remote_code=True)
    model.eval().to(DEV)
    for pp in model.parameters(): pp.requires_grad_(False)
    d=model.config.hidden_size
    ds=make_ds(tok,n=args.n,sl=args.ctx+1)
    H=embed_bank(model,ds,args.layer,DEV,ctx=args.ctx)
    print(f"H={H.shape} d={d}")
    S=sample_margin_cov(H,args.n_pairs)
    print(f"tr(S)={S.trace().item():.4f}")
    M=d*S-torch.eye(d)
    ev=torch.linalg.eigvalsh((M+M.T)/2)
    eps_iso=float(ev.abs().max())
    print(f"eps_iso = ||d*S - I||_op = {eps_iso:.4f}")
    rows=[]
    for k in ks:
        P_R=random_projector(d,k,seed=0)
        tr_R=float((P_R*S).sum())
        expected=k/d
        band=(k/d)*eps_iso
        rows.append({"k":k,"kind":"random","tr_PS":tr_R,"expected":expected,"band":band,"within":abs(tr_R-expected)<=band+1e-6})
    out={"model":args.model,"layer":args.layer,"d":d,"n":args.n,"n_pairs":args.n_pairs,"eps_iso":eps_iso,"tr_S":float(S.trace()),"rows":rows,"elapsed_s":time.time()-t0}
    slug=args.model.replace("/","_")
    with open(OUT/f"isotropy_{slug}.json","w") as f: json.dump(out,f,indent=2)
    print(f"saved -> {OUT}/isotropy_{slug}.json")
    for r in rows:
        print(f"  k={r['k']:4d} random tr(PS)={r['tr_PS']:.4f} expected={r['expected']:.4f} band=±{r['band']:.4f} within={r['within']}")

if __name__=="__main__": main()

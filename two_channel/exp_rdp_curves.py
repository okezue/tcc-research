#!/usr/bin/env python3
"""Compute empirical Renyi-DP curves for each defense mechanism by building an
adjacency set A from hidden-state pairs and applying the accountant.

Runs locally using the saved Mahalanobis JSONs (which carry tr(F), tr(S))
plus re-embedded H_test from the same model/layer.

For each (model, mechanism, sigma) produces (eps, KL) curves; outputs one JSON
per model for figure 6 in the paper.
"""
import os,sys,json,argparse
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from pathlib import Path
from two_channel.exp_optimal_defense import make_ds,embed_bank,compute_fisher_avg,compute_id_cov,gen_eigendecomp
from two_channel.mahalanobis_defense import solve_mahalanobis_optimal
from two_channel.rdp_accountant import eps_delta
from two_channel.adjacency_builder import nearest_neighbors,random_neighbors

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/rdp")
OUT.mkdir(parents=True,exist_ok=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--n_cal",type=int,default=200)
    p.add_argument("--n_bank",type=int,default=1000)
    p.add_argument("--sigmas",default="0.5,1.0,2.0,5.0")
    p.add_argument("--delta",type=float,default=1e-6)
    p.add_argument("--eta_ratio",type=float,default=1e-3)
    args=p.parse_args()
    args.sigmas=[float(x) for x in args.sigmas.split(",")]
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    is_gpt2="gpt2" in args.model.lower()
    dtype=torch.float32 if is_gpt2 else torch.float16
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype,output_hidden_states=True,trust_remote_code=True)
    model.eval().to(DEV)
    for pp in model.parameters(): pp.requires_grad_(False)
    d=model.config.hidden_size
    ctx=32
    ds=make_ds(tok,n=args.n_cal+args.n_bank,sl=ctx+1)
    print(f"[rdp] {args.model} L{args.layer} d={d}")
    F=compute_fisher_avg(model,ds[:args.n_cal],args.layer,DEV,n_cal=args.n_cal,ctx=ctx)
    H=embed_bank(model,ds[args.n_cal:args.n_cal+args.n_bank],args.layer,DEV,ctx=ctx)
    S=compute_id_cov(H,n_pairs=min(len(H),2000))
    A_nn=nearest_neighbors(H,k=min(64,len(H)-1))
    A_rand=random_neighbors(H,k=64)
    Deltas=torch.cat([A_nn,A_rand],0)[:500]
    print(f"  |A|={Deltas.shape[0]} tr(F)={F.trace().item():.3f} tr(S)={S.trace().item():.3f}")
    rows=[]
    for sigma in args.sigmas:
        kappa=sigma*sigma*F.trace().item()
        Sigma_iso=(sigma**2)*torch.eye(d)
        res=solve_mahalanobis_optimal(F,S,kappa,eta_ratio=args.eta_ratio)
        Sigma_mah=res["Sigma_star"]
        r_iso=eps_delta(Deltas,Sigma_iso,delta=args.delta)
        r_mah=eps_delta(Deltas,Sigma_mah,delta=args.delta)
        kl_iso=0.5*(F*Sigma_iso).sum().item()
        kl_mah=0.5*(F*Sigma_mah).sum().item()
        rows.append({"sigma":sigma,"kappa":kappa,
            "iso_eps":r_iso["eps"],"iso_kl":kl_iso,
            "mah_eps":r_mah["eps"],"mah_kl":kl_mah,
            "alpha_iso":r_iso["alpha_star"],"alpha_mah":r_mah["alpha_star"]})
        print(f"  sigma={sigma}  iso eps={r_iso['eps']:.2f}  mah eps={r_mah['eps']:.2f}  kl_iso={kl_iso:.2f}  kl_mah={kl_mah:.2f}")
    out={"model":args.model,"layer":args.layer,"d":d,"delta":args.delta,"eta_ratio":args.eta_ratio,"rows":rows}
    slug=args.model.replace("/","_")
    with open(OUT/f"rdp_{slug}.json","w") as f: json.dump(out,f,indent=2)
    print(f"saved -> {OUT}/rdp_{slug}.json")

if __name__=="__main__": main()

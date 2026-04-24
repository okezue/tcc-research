#!/usr/bin/env python3
"""Run SDP worst-case DP covariance on GPT-2 to verify closed-form Mahalanobis is comparable."""
import os,sys,json,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from pathlib import Path
from two_channel.exp_optimal_defense import make_ds,embed_bank,compute_fisher_avg,compute_id_cov
from two_channel.mahalanobis_defense import solve_mahalanobis_optimal
from two_channel.sdp_worst_case import sdp_worst_case
from two_channel.adjacency_builder import nearest_neighbors,random_neighbors
from two_channel.rdp_accountant import eps_delta

DEV="cuda" if torch.cuda.is_available() else "cpu"

def main():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model_name="openai-community/gpt2"
    tok=AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float32,output_hidden_states=True).eval().to(DEV)
    for pp in model.parameters(): pp.requires_grad_(False)
    d=model.config.hidden_size

    ds=make_ds(tok,n=1000,sl=33)
    F=compute_fisher_avg(model,ds[:200],6,DEV,n_cal=200,ctx=32)
    H=embed_bank(model,ds[200:1200],6,DEV,ctx=32)
    S=compute_id_cov(H,n_pairs=2000)

    # small adjacency (SDP scales poorly)
    A=torch.cat([nearest_neighbors(H,k=20),random_neighbors(H,k=10)],0)[:40]
    print(f"|A|={A.shape[0]}")

    kappa=float(F.trace())*1.0  # σ=1 equivalent
    print("Running SDP (reduced basis, rank r=64)...")
    t0=time.time()
    Sigma_sdp,t_star,r_eff=sdp_worst_case(F,S,A,kappa,r=64,eta=1e-3*F.trace().item()/d)
    elapsed=time.time()-t0
    print(f"SDP done in {elapsed:.0f}s. reduced_rank={r_eff} t*={t_star:.3f}")

    # Compare to closed-form Mahalanobis
    mh=solve_mahalanobis_optimal(F,S,kappa,eta_ratio=1e-3)
    Sigma_mah=mh["Sigma_star"]

    r_sdp=eps_delta(A,Sigma_sdp,delta=1e-6)
    r_mah=eps_delta(A,Sigma_mah,delta=1e-6)
    r_iso=eps_delta(A,torch.eye(d)*(kappa/F.trace().item()),delta=1e-6)

    out={"model":model_name,"layer":6,"d":d,"r_eff":r_eff,"|A|":A.shape[0],
         "t_star_sdp":t_star,"elapsed_s":elapsed,
         "eps_sdp":r_sdp["eps"],"eps_mah":r_mah["eps"],"eps_iso":r_iso["eps"]}
    Path("artifacts/sdp").mkdir(parents=True,exist_ok=True)
    with open("artifacts/sdp/sdp_gpt2.json","w") as f: json.dump(out,f,indent=2)
    print(f"eps(δ=1e-6): SDP={r_sdp['eps']:.2f} MahOpt={r_mah['eps']:.2f} iso={r_iso['eps']:.2f}")

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Pass 1 experiment: Mahalanobis-optimal defense + defense-adaptive attackers.

For a given model and layer, compute:
  - Fisher F (empirical, 10 top-k sampled-label gradients per prefix)
  - Margin-direction covariance S from nearest-neighbor pairs
  - Four defenses: isotropic, complement, generalized-eigen, Mahalanobis-optimal
  - Three attackers per defense:
        L2 in full space
        subspace P_I
        defense-adaptive Mahalanobis Sigma^{-1}
  - G_Euc and G_Mah scalars
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import json,time,argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from two_channel.mahalanobis_defense import (
    solve_mahalanobis_optimal, gen_eigen_gain, sample_gaussian_with_cov)
from two_channel.mahalanobis_attacker import (
    mahalanobis_retrieval, tune_tau, l2_retrieval, subspace_retrieval)
from two_channel.exp_optimal_defense import (
    make_ds, get_layer_block, compute_fisher_avg, compute_id_cov,
    embed_bank, gen_eigendecomp, measure_kl)

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/mahalanobis")
OUT.mkdir(parents=True,exist_ok=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--n_cal",type=int,default=500)
    p.add_argument("--n_bank",type=int,default=1000)
    p.add_argument("--n_query",type=int,default=200)
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    mname=args.model
    dev=args.device
    t0=time.time()
    print("="*60+f"\nMahalanobis defense: {mname} L{args.layer} k={args.k}\n"+"="*60)

    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(mname,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    is_gpt2="gpt2" in mname.lower()
    dtype=torch.float32 if is_gpt2 else torch.float16
    model=AutoModelForCausalLM.from_pretrained(mname,torch_dtype=dtype,output_hidden_states=True,trust_remote_code=True)
    model.eval().to(dev)
    for pp in model.parameters(): pp.requires_grad_(False)
    d=model.config.hidden_size
    layer=args.layer
    k=args.k
    ctx=32

    print(f"d={d} layer={layer} k={k}")

    total=args.n_cal+args.n_bank+args.n_query+200
    ds=make_ds(tok,n=total,sl=ctx+1)
    ds_cal=ds[:args.n_cal]
    ds_bank=ds[args.n_cal:args.n_cal+args.n_bank]
    ds_val=ds[args.n_cal+args.n_bank:args.n_cal+args.n_bank+200]
    ds_test=ds[args.n_cal+args.n_bank+200:args.n_cal+args.n_bank+200+args.n_query]

    print("\n--- Computing Fisher (empirical) ---")
    F_mat=compute_fisher_avg(model,ds_cal,layer,dev,n_cal=args.n_cal,ctx=ctx)
    print(f"tr(F)={F_mat.trace().item():.4f}")

    print("\n--- Embedding bank / val / test ---")
    H_bank=embed_bank(model,ds_bank,layer,dev,ctx=ctx)
    H_val=embed_bank(model,ds_val,layer,dev,ctx=ctx)
    H_test=embed_bank(model,ds_test,layer,dev,ctx=ctx)
    print(f"bank={H_bank.shape} val={H_val.shape} test={H_test.shape}")

    print("\n--- Computing S from NN pairs ---")
    S=compute_id_cov(H_bank,n_pairs=min(len(H_bank),2000))
    print(f"tr(S)={S.trace().item():.4f}")

    print("\n--- Gen-eigen decomposition (for current defense + P_B / P_I) ---")
    lambdas,V_gen,_,_=gen_eigendecomp(S,F_mat)
    V_gen_norm=V_gen.clone()
    for i in range(V_gen_norm.shape[1]):
        fv=(V_gen_norm[:,i].double()@F_mat.double()@V_gen_norm[:,i].double()).sqrt().clamp(min=1e-8)
        V_gen_norm[:,i]=V_gen_norm[:,i]/fv.float()

    cov=H_bank.T@H_bank/H_bank.shape[0]
    evals_cov,evecs_cov=torch.linalg.eigh(cov)
    idx_cov=evals_cov.argsort(descending=True)
    U_B=evecs_cov[:,idx_cov][:,:k]
    V_id=torch.eye(d)-U_B@U_B.T

    print("\n--- G_Euc and G_Mah scalars ---")
    kappa_trial=1.0
    mah=solve_mahalanobis_optimal(F_mat,S,kappa_trial)
    euc=gen_eigen_gain(F_mat,S,k)
    print(f"G_Euc (top-{k}) = {euc['G_Euc']:.2f}")
    print(f"G_Mah (trace-based) = {mah['G_Mah']:.2f}")

    print("\n--- Building defense mechanisms ---")

    def noise_isotropic(h,sigma):
        return h+torch.randn_like(h)*sigma
    def Sigma_iso(sigma):
        return (sigma**2)*torch.eye(d)

    def noise_complement(h,sigma,U=U_B):
        z=torch.randn_like(h)*sigma
        return h+(z-z@U@U.T)
    def Sigma_complement(sigma,U=U_B):
        P_I=torch.eye(d)-U@U.T
        return (sigma**2)*P_I

    def noise_gen_eigen(h,sigma,V=V_gen_norm,k_noise=k):
        z=torch.zeros(d)
        z[:k_noise]=torch.randn(k_noise)*sigma
        return h+V@z
    def Sigma_gen_eigen(sigma,V=V_gen_norm,k_noise=k):
        V_k=V[:,:k_noise]
        return (sigma**2)*(V_k@V_k.T)

    F_diag=F_mat.diagonal().clamp(min=1e-8)
    F_inv_diag=(1.0/F_diag)
    F_inv_diag_scale=(F_inv_diag*F_diag).sum()/F_inv_diag.sum()
    def noise_fisher_diag(h,sigma,fd=F_inv_diag):
        return h+torch.randn_like(h)*fd.sqrt()*sigma
    def Sigma_fisher_diag(sigma,fd=F_inv_diag):
        return (sigma**2)*torch.diag(fd)

    g=torch.Generator().manual_seed(7)
    R=torch.randn(d,k,generator=g)
    R,_=torch.linalg.qr(R)
    P_R=R@R.T
    def noise_randproj(h,sigma,P=P_R):
        z=torch.randn_like(h)*sigma
        return h+(P@z.reshape(-1)).reshape(h.shape) if h.dim()==1 else h+(z@P.T)
    def Sigma_randproj(sigma,P=P_R):
        return (sigma**2)*P

    def quantizer(b):
        def q(h,sigma_unused=None,b=b):
            amax=h.abs().max().item()
            if amax<1e-8: return h.clone()
            step=amax/(2**(b-1)-1)
            return torch.round(h/step)*step
        return q
    def Sigma_quant(b,d_=d):
        return (1.0/(3*(2**(2*b)-1)))*torch.eye(d_)

    def noise_dropout(h,sigma,p=0.1):
        mask=(torch.rand_like(h)>p).float()
        return h*mask/(1-p)
    def Sigma_dropout(sigma,p=0.1,d_=d):
        return (p/(1-p))*torch.eye(d_)

    def noise_bproj_dp(h,sigma,U=U_B):
        z=torch.randn_like(h)*sigma
        return h+(z@U@U.T)
    def Sigma_bproj_dp(sigma,U=U_B):
        return (sigma**2)*(U@U.T)

    def build_mahalanobis_mechanism(kappa,eta_ratio=1e-3):
        res=solve_mahalanobis_optimal(F_mat,S,kappa,eta_ratio=eta_ratio)
        Sigma=res["Sigma_star"]
        evals,evecs=torch.linalg.eigh((Sigma+Sigma.T)/2)
        evals=evals.clamp(min=0)
        L=evecs@torch.diag(evals.sqrt())
        def noise_mah(h,sigma_unused=None,L=L):
            z=torch.randn(d)
            return h+L@z
        return Sigma,noise_mah,res

    sigmas=[0.1,0.3,0.5,1.0,2.0,5.0]
    results=[]
    H_all_test=torch.cat([H_bank,H_test],dim=0)
    test_idx=list(range(len(H_bank),len(H_all_test)))
    H_all_val=torch.cat([H_bank,H_val],dim=0)
    val_idx=list(range(len(H_bank),len(H_all_val)))

    mechanisms_by_sigma={}
    for sigma in sigmas:
        mechanisms_by_sigma[sigma]={
            "isotropic":(noise_isotropic,Sigma_iso(sigma)),
            "complement":(noise_complement,Sigma_complement(sigma)),
            "gen_eigen":(noise_gen_eigen,Sigma_gen_eigen(sigma)),
            "fisher_diag_inv":(noise_fisher_diag,Sigma_fisher_diag(sigma)),
            "random_proj":(noise_randproj,Sigma_randproj(sigma)),
            "dropout":(noise_dropout,Sigma_dropout(sigma)),
            "bproj_dp":(noise_bproj_dp,Sigma_bproj_dp(sigma)),
        }
    quant_entries={}
    for b in [8,4,2]:
        quant_entries[f"quant_b{b}"]=(quantizer(b),Sigma_quant(b))
    print("\n--- Calibrating Mahalanobis mechanism to matched tr(F*Sigma_iso) ---")
    for sigma in sigmas:
        kappa=float((F_mat*Sigma_iso(sigma)).sum())
        Sigma_mah,noise_mah,res=build_mahalanobis_mechanism(kappa,eta_ratio=1e-3)
        mechanisms_by_sigma[sigma]["mahalanobis_opt"]=(noise_mah,Sigma_mah)
        if sigma==1.0:
            print(f"  sigma=1.0: G_Mah={res['G_Mah']:.2f} tr(Sigma_mah)={Sigma_mah.trace().item():.2f}")

    print("\n--- DEFENSE x ATTACKER SWEEP ---")
    sweep_iter=[]
    for sigma in sigmas:
        for mech_name,(mech_fn,Sigma) in mechanisms_by_sigma[sigma].items():
            sweep_iter.append((sigma,mech_name,mech_fn,Sigma))
    for mech_name,(mech_fn,Sigma) in quant_entries.items():
        sweep_iter.append((0.0,mech_name,mech_fn,Sigma))
    for sigma,mech_name,mech_fn,Sigma in sweep_iter:
            torch.manual_seed(42)
            H_noisy=torch.stack([mech_fn(H_test[i],sigma) for i in range(len(H_test))])

            r_l2=l2_retrieval(H_noisy,H_all_test,test_idx)
            r_id=subspace_retrieval(H_noisy,H_all_test,V_id,test_idx)

            tau_best,_=tune_tau(H_val,H_all_val,Sigma,val_idx)
            r_mah=mahalanobis_retrieval(H_noisy,H_all_test,Sigma,tau_best,test_idx)

            best_top1=max(r_l2["top1"],r_id["top1"],r_mah["top1"])

            if sigma>0:
                mean_kl,t1=measure_kl(model,ds_test,layer,dev,lambda h,s=sigma,fn=mech_fn:fn(h,s),n_eval=100,ctx=ctx)
            else:
                mean_kl=0;t1=1.0

            entry={
                "mechanism":mech_name,"sigma":sigma,"mean_kl":mean_kl,"t1_agree":t1,
                "attack_L2_top1":r_l2["top1"],"attack_L2_mrr":r_l2["mrr"],
                "attack_Id_top1":r_id["top1"],"attack_Id_mrr":r_id["mrr"],
                "attack_Mah_top1":r_mah["top1"],"attack_Mah_mrr":r_mah["mrr"],
                "attack_Mah_tau":float(tau_best),
                "best_attack_top1":best_top1,
            }
            results.append(entry)
            print(f"    {mech_name:>18s} kl={mean_kl:.3f} L2={r_l2['top1']:.3f} Id={r_id['top1']:.3f} Mah={r_mah['top1']:.3f} best={best_top1:.3f}")

    output={
        "model":mname,"layer":layer,"k":k,"d":d,"ctx":ctx,
        "n_cal":args.n_cal,"n_bank":args.n_bank,"n_query":args.n_query,
        "fisher_trace":float(F_mat.trace()),
        "id_cov_trace":float(S.trace()),
        "G_Euc":euc["G_Euc"],"G_Mah":mah["G_Mah"],
        "top_gen_eigenvalues":euc["top_eigvals"],
        "tr_C_half":mah["tr_C_half"],
        "defense_sweep":results,
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s":time.time()-t0,
    }
    slug=mname.replace("/","_")
    with open(OUT/f"mahalanobis_{slug}.json","w") as f:
        json.dump(output,f,indent=2)
    print(f"\nResults saved to {OUT}/mahalanobis_{slug}.json")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__=="__main__":
    main()

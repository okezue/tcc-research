#!/usr/bin/env python3
"""Matched-epsilon calibration sweep.

For each model at target eps in {1,3,8,16}, calibrate scalar c such that
  eps_delta(c * Sigma) = eps_target  for each defense mechanism,
then evaluate retrieval attack success at the calibrated scale.
Produces a table: (model, mechanism, eps_target, scalar, attack_success).
"""
import os,sys,json,argparse,time
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from pathlib import Path
from two_channel.exp_optimal_defense import make_ds,embed_bank,compute_fisher_avg,compute_id_cov,gen_eigendecomp,measure_kl
from two_channel.mahalanobis_defense import solve_mahalanobis_optimal
from two_channel.mahalanobis_attacker import mahalanobis_retrieval,tune_tau,l2_retrieval
from two_channel.mahalanobis_attacker import _EIGH_CACHE
from two_channel.rdp_accountant import eps_delta,calibrate_scalar_to_eps
from two_channel.adjacency_builder import random_neighbors,nearest_neighbors

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/matched_eps")
OUT.mkdir(parents=True,exist_ok=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--n_cal",type=int,default=300)
    p.add_argument("--n_bank",type=int,default=2000)
    p.add_argument("--n_query",type=int,default=500)
    p.add_argument("--eps_targets",default="1,3,8,16")
    p.add_argument("--delta",type=float,default=1e-6)
    p.add_argument("--eta_ratio",type=float,default=1e-3)
    args=p.parse_args()
    eps_targets=[float(x) for x in args.eps_targets.split(",")]
    t0=time.time()

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

    total=args.n_cal+args.n_bank+args.n_query+200
    ds=make_ds(tok,n=total,sl=ctx+1)
    ds_cal=ds[:args.n_cal]
    ds_bank=ds[args.n_cal:args.n_cal+args.n_bank]
    ds_val=ds[args.n_cal+args.n_bank:args.n_cal+args.n_bank+200]
    ds_test=ds[args.n_cal+args.n_bank+200:args.n_cal+args.n_bank+200+args.n_query]

    print(f"[matched_eps] {args.model} L{args.layer} d={d}")
    F=compute_fisher_avg(model,ds_cal,args.layer,DEV,n_cal=args.n_cal,ctx=ctx)
    H_bank=embed_bank(model,ds_bank,args.layer,DEV,ctx=ctx)
    H_val=embed_bank(model,ds_val,args.layer,DEV,ctx=ctx)
    H_test=embed_bank(model,ds_test,args.layer,DEV,ctx=ctx)
    S=compute_id_cov(H_bank,n_pairs=min(len(H_bank),2000))

    # adjacency set
    A=torch.cat([nearest_neighbors(H_bank,k=min(64,len(H_bank)-1)),
                 random_neighbors(H_bank,k=64)],0)[:500]
    print(f"|A|={A.shape[0]}")

    F_diag_inv=(1.0/F.diagonal().clamp(min=1e-8))
    mh=solve_mahalanobis_optimal(F,S,1.0,eta_ratio=args.eta_ratio)
    # base Sigmas at sigma=1 (will scale)
    sigmas_base={
        "isotropic":torch.eye(d),
        "fisher_diag_inv":torch.diag(F_diag_inv),
        "mahalanobis_opt":mh["Sigma_star"],
    }

    H_all=torch.cat([H_bank,H_test],dim=0)
    test_idx=list(range(len(H_bank),len(H_all)))
    H_all_val=torch.cat([H_bank,H_val],dim=0)
    val_idx=list(range(len(H_bank),len(H_all_val)))

    rows=[]
    for mech,S0 in sigmas_base.items():
        for eps_t in eps_targets:
            c=calibrate_scalar_to_eps(A,S0,eps_target=eps_t,delta=args.delta,eta_ratio=args.eta_ratio)
            if c is None:
                rows.append({"mech":mech,"eps_target":eps_t,"c":None,"err":"infeasible"})
                continue
            Sigma=S0*c
            # sample
            ev,U=torch.linalg.eigh((Sigma+Sigma.T)/2)
            L=U@torch.diag(ev.clamp(min=0).sqrt())
            torch.manual_seed(42)
            z=torch.randn(len(H_test),d)
            H_noisy=H_test+z@L.T
            _EIGH_CACHE.clear()
            tau_best,_=tune_tau(H_val,H_all_val,Sigma,val_idx)
            r_mah=mahalanobis_retrieval(H_noisy,H_all,Sigma,tau_best,test_idx)
            r_l2=l2_retrieval(H_noisy,H_all,test_idx)
            best=max(r_l2["top1"],r_mah["top1"])
            # measure KL
            def mk_fn(Lmat=L):
                def fn(h):
                    zz=torch.randn(d,device=h.device)
                    return h+(Lmat.to(h.device)@zz).to(h.dtype)
                return fn
            kl,_=measure_kl(model,ds_test,args.layer,DEV,mk_fn(),n_eval=64,ctx=ctx)
            rows.append({"mech":mech,"eps_target":eps_t,"c":float(c),"kl":float(kl),
                         "l2_top1":r_l2["top1"],"mah_top1":r_mah["top1"],"best_top1":best})
            print(f"  {mech:>16s} eps={eps_t} c={c:.3f} kl={kl:.2f} best={best:.3f}")
    out={"model":args.model,"layer":args.layer,"d":d,"delta":args.delta,"rows":rows,"elapsed_s":time.time()-t0}
    slug=args.model.replace("/","_")
    with open(OUT/f"matched_eps_{slug}.json","w") as f: json.dump(out,f,indent=2)
    print(f"saved -> {OUT}/matched_eps_{slug}.json")

if __name__=="__main__": main()

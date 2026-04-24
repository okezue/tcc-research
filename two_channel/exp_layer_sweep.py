#!/usr/bin/env python3
"""Layer sweep: for each layer, compute G_Euc, G_Mah, and observed adaptive gain
under two fixed noise levels. Produces (model,layer) points for the oral figure 5.
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch,json,time,argparse
from pathlib import Path
from two_channel.exp_optimal_defense import make_ds,compute_fisher_avg,compute_id_cov,embed_bank,gen_eigendecomp,measure_kl
from two_channel.mahalanobis_defense import solve_mahalanobis_optimal,gen_eigen_gain
from two_channel.mahalanobis_attacker import mahalanobis_retrieval,tune_tau,l2_retrieval
from two_channel.mahalanobis_attacker import _EIGH_CACHE

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/layer_sweep")
OUT.mkdir(parents=True,exist_ok=True)

def sweep_layer(model,tok,layer,args):
    ctx=32
    total=args.n_cal+args.n_bank+args.n_query+200
    ds=make_ds(tok,n=total,sl=ctx+1)
    ds_cal=ds[:args.n_cal]
    ds_bank=ds[args.n_cal:args.n_cal+args.n_bank]
    ds_val=ds[args.n_cal+args.n_bank:args.n_cal+args.n_bank+200]
    ds_test=ds[args.n_cal+args.n_bank+200:args.n_cal+args.n_bank+200+args.n_query]
    print(f"  fisher...")
    F=compute_fisher_avg(model,ds_cal,layer,DEV,n_cal=args.n_cal,ctx=ctx)
    print(f"  tr(F)={F.trace().item():.3f}")
    print(f"  embed bank/val/test...")
    H_bank=embed_bank(model,ds_bank,layer,DEV,ctx=ctx)
    H_val=embed_bank(model,ds_val,layer,DEV,ctx=ctx)
    H_test=embed_bank(model,ds_test,layer,DEV,ctx=ctx)
    S=compute_id_cov(H_bank,n_pairs=min(len(H_bank),2000))
    print(f"  tr(S)={S.trace().item():.3f}")
    gm=solve_mahalanobis_optimal(F,S,1.0)
    ge=gen_eigen_gain(F,S,args.k)
    G_Euc=ge["G_Euc"];G_Mah=gm["G_Mah"]
    d=F.shape[0]
    rows=[]
    H_all=torch.cat([H_bank,H_test],dim=0)
    test_idx=list(range(len(H_bank),len(H_all)))
    H_all_val=torch.cat([H_bank,H_val],dim=0)
    val_idx=list(range(len(H_bank),len(H_all_val)))
    for sigma in args.sigmas:
        kappa=sigma*sigma*F.trace().item()
        Sigma_iso=(sigma**2)*torch.eye(d)
        mh=solve_mahalanobis_optimal(F,S,kappa,eta_ratio=1e-3)
        Sigma_mah=mh["Sigma_star"]
        torch.manual_seed(42)
        H_noisy_iso=H_test+torch.randn_like(H_test)*sigma
        ev,U=torch.linalg.eigh((Sigma_mah+Sigma_mah.T)/2)
        L=U@torch.diag(ev.clamp(min=0).sqrt())
        torch.manual_seed(42)
        z=torch.randn(len(H_test),d)
        H_noisy_mah=H_test+z@L.T
        _EIGH_CACHE.clear()
        tau_i,_=tune_tau(H_val,H_all_val,Sigma_iso,val_idx)
        r_iso_mah=mahalanobis_retrieval(H_noisy_iso,H_all,Sigma_iso,tau_i,test_idx)
        r_iso_l2=l2_retrieval(H_noisy_iso,H_all,test_idx)
        _EIGH_CACHE.clear()
        tau_m,_=tune_tau(H_val,H_all_val,Sigma_mah,val_idx)
        r_mah_mah=mahalanobis_retrieval(H_noisy_mah,H_all,Sigma_mah,tau_m,test_idx)
        r_mah_l2=l2_retrieval(H_noisy_mah,H_all,test_idx)
        def iso_noise(h,s=sigma): return h+torch.randn_like(h)*s
        def mah_noise(h,L=L):
            z=torch.randn(d,device=h.device)
            return h+(L.to(h.device)@z).to(h.dtype)
        kl_iso,_=measure_kl(model,ds_test,layer,DEV,iso_noise,n_eval=64,ctx=ctx)
        kl_mah,_=measure_kl(model,ds_test,layer,DEV,mah_noise,n_eval=64,ctx=ctx)
        rows.append({
            "sigma":sigma,"kappa":kappa,
            "iso_kl":kl_iso,"iso_mah_top1":r_iso_mah["top1"],"iso_l2_top1":r_iso_l2["top1"],
            "mah_kl":kl_mah,"mah_mah_top1":r_mah_mah["top1"],"mah_l2_top1":r_mah_l2["top1"],
            "observed_gain":max(r_iso_mah["top1"],r_iso_l2["top1"])/max(1e-3,max(r_mah_mah["top1"],r_mah_l2["top1"])),
        })
        print(f"  sigma={sigma} kl_iso={kl_iso:.2f} iso_best={max(r_iso_mah['top1'],r_iso_l2['top1']):.3f} kl_mah={kl_mah:.2f} mah_best={max(r_mah_mah['top1'],r_mah_l2['top1']):.3f}")
    return {"layer":layer,"d":d,"tr_F":float(F.trace()),"tr_S":float(S.trace()),"G_Euc":G_Euc,"G_Mah":G_Mah,"sigmas":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layers",default="2,4,6,8,10")
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--n_cal",type=int,default=200)
    p.add_argument("--n_bank",type=int,default=2000)
    p.add_argument("--n_query",type=int,default=300)
    p.add_argument("--sigmas",default="1.0,3.0")
    args=p.parse_args()
    args.layers=[int(x) for x in args.layers.split(",")]
    args.sigmas=[float(x) for x in args.sigmas.split(",")]
    t0=time.time()
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    is_gpt2="gpt2" in args.model.lower()
    dtype=torch.float32 if is_gpt2 else torch.float16
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype,output_hidden_states=True,trust_remote_code=True)
    model.eval().to(DEV)
    for pp in model.parameters(): pp.requires_grad_(False)
    out={"model":args.model,"layers":[]}
    for layer in args.layers:
        print(f"=== layer {layer} ===")
        r=sweep_layer(model,tok,layer,args)
        out["layers"].append(r)
        slug=args.model.replace("/","_")
        with open(OUT/f"layer_sweep_{slug}.json","w") as f: json.dump(out,f,indent=2)
    out["elapsed_s"]=time.time()-t0
    slug=args.model.replace("/","_")
    with open(OUT/f"layer_sweep_{slug}.json","w") as f: json.dump(out,f,indent=2)
    print(f"saved -> {OUT}/layer_sweep_{slug}.json ({out['elapsed_s']:.0f}s)")

if __name__=="__main__": main()

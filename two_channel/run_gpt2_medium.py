#!/usr/bin/env python3
"""GPT-2 Medium scaling confirmation: subspace + utility + leakage."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch,json,time
from pathlib import Path

DEV="cuda" if torch.cuda.is_available() else "mps"
MODEL="openai-community/gpt2-medium"
OUT=Path("artifacts/gpt2_medium")
OUT.mkdir(parents=True,exist_ok=True)

def step1():
    print("="*60+"\nGPT2-MEDIUM: Subspace computation\n"+"="*60)
    from two_channel.compute_subspace import load_model,make_calibration_dataset,compute_gradient_covariance,save_subspace
    model,tok=load_model(MODEL,DEV)
    ds=make_calibration_dataset(tok,n=3000,seq_len=33)
    for li in [12,-1]:
        res=compute_gradient_covariance(model,tok,li,ds,prefix_len=32,device=DEV,max_samples=3000)
        save_subspace(res,OUT/"subspace",MODEL,res["layer_idx"])
        evals=res["grad_eigenvalues"]
        total=evals.sum()
        print(f"Layer {res['layer_idx']}:")
        for k in [32,64,128,256]:
            print(f"  top-{k} energy: {evals[:k].sum()/total:.4f}")
    del model; torch.cuda.empty_cache() if DEV=="cuda" else None

def step2():
    print("="*60+"\nGPT2-MEDIUM: Utility evaluation\n"+"="*60)
    from two_channel.eval_utility import run_utility_grid
    r,base=run_utility_grid(
        MODEL,layers=[12,-1],k_values=[64,256],
        bits_values=[32,8],sigma_values=[0.0],
        device=DEV,prefix_len=32,n_eval=1000,n_cal=500,
        subspace_dir=str(OUT/"subspace")
    )
    (OUT/"results").mkdir(parents=True,exist_ok=True)
    with open(OUT/"results/utility.json","w") as f:
        json.dump({"baseline_ppl":base,"results":r},f,indent=2)
    print(f"Baseline PPL: {base:.2f}, {len(r)} configs")

def step3():
    print("="*60+"\nGPT2-MEDIUM: Leakage evaluation\n"+"="*60)
    from two_channel.eval_leakage import eval_leakage_grid
    r=eval_leakage_grid(
        MODEL,layers=[12,-1],k_values=[64,256],
        bits_values=[8],device=DEV,prefix_len=16,
        n_prefixes=50,batch_size=1024,
        subspace_dir=str(OUT/"subspace")
    )
    with open(OUT/"results/leakage.json","w") as f:
        json.dump({"results":r},f,indent=2)

def step4():
    print("="*60+"\nGPT2-MEDIUM: Plots\n"+"="*60)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    with open(OUT/"results/utility.json") as f: u=json.load(f)
    with open(OUT/"results/leakage.json") as f: l=json.load(f)
    COLORS={'behavior':'#e74c3c','identity':'#27ae60','random':'#2980b9'}

    (OUT/"plots").mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,5))

    ax=axes[0]
    for mode in ['behavior','identity','random']:
        pts=[(r['k'],np.log10(max(r['dppl'],0.01))) for r in u['results']
             if r['bits']==32 and r['mode']==mode and r['sigma']==0.0]
        if pts:
            pts.sort(); ks,dp=zip(*pts)
            ax.plot(ks,dp,color=COLORS[mode],marker='o',label=mode,linewidth=2)
    ax.set_xlabel('k'); ax.set_ylabel('log10(dPPL)')
    ax.set_title('GPT-2 Medium: Utility vs k (fp32)'); ax.legend(); ax.grid(alpha=0.3)

    ax=axes[1]
    for mode in ['behavior','identity','random']:
        pts=[(r['k'],r['margin_median']) for r in l['results']
             if r['bits']==8 and r['mode']==mode]
        if pts:
            pts.sort(); ks,m=zip(*pts)
            ax.plot(ks,m,color=COLORS[mode],marker='s',label=mode,linewidth=2)
    ax.set_xlabel('k'); ax.set_ylabel('Median margin')
    ax.set_title('GPT-2 Medium: Margin vs k (8-bit)'); ax.legend(); ax.grid(alpha=0.3)

    ax=axes[2]
    for mode in ['behavior','identity','random']:
        xs,ys=[],[]
        for lr in l['results']:
            if lr['mode']!=mode: continue
            match=[ur for ur in u['results'] if ur['k']==lr['k'] and ur['mode']==mode
                   and ur['bits']==lr['bits'] and ur['sigma']==0.0 and ur['layer']==lr['layer']]
            if match:
                xs.append(np.log10(max(match[0]['dppl'],0.01)))
                ys.append(lr['margin_median'])
        if xs: ax.scatter(xs,ys,color=COLORS[mode],marker='o',label=mode,s=60)
    ax.set_xlabel('log10(dPPL)'); ax.set_ylabel('Margin')
    ax.set_title('GPT-2 Medium: Privacy-Utility Frontier'); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('GPT-2 Medium Scaling Confirmation',fontsize=14,fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/"plots/gpt2_medium_summary.png",dpi=150)
    plt.close(fig)
    print(f"Plots saved to {OUT/'plots'}")

if __name__=="__main__":
    t0=time.time()
    for fn in [step1,step2,step3,step4]:
        try: fn()
        except Exception as e:
            print(f"Failed: {e}")
            import traceback; traceback.print_exc()
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

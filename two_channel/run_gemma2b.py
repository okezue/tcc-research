#!/usr/bin/env python3
"""Gemma-2B scaling confirmation: subspace + utility + leakage."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch,json,time
from pathlib import Path

DEV="cuda" if torch.cuda.is_available() else "mps"
MODEL="google/gemma-2-2b"
OUT=Path("artifacts/gemma2b")
OUT.mkdir(parents=True,exist_ok=True)

def step1():
    print("="*60+"\nGEMMA-2B: Subspace computation\n"+"="*60)
    from two_channel.compute_subspace import load_model,compute_gradient_covariance,save_subspace
    from transformers import AutoTokenizer
    from datasets import load_dataset

    model,tok=load_model(MODEL,DEV)
    n_layers=model.config.num_hidden_layers
    mid=n_layers//2
    print(f"Gemma-2B: {n_layers} layers, d_model={model.config.hidden_size}, mid={mid}")

    torch.manual_seed(42)
    ds_raw=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    all_toks=[]
    for row in ds_raw:
        txt=row["text"].strip()
        if len(txt)<50: continue
        ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=33:
            start=torch.randint(0,len(ids)-33+1,(1,)).item()
            all_toks.append(torch.tensor(ids[start:start+33],dtype=torch.long))
            if len(all_toks)>=2000: break
    print(f"Dataset: {len(all_toks)} sequences")

    for li in [mid,-1]:
        res=compute_gradient_covariance(model,tok,li,all_toks,prefix_len=32,device=DEV,max_samples=2000)
        save_subspace(res,OUT/"subspace",MODEL,res["layer_idx"])
        evals=res["grad_eigenvalues"]
        total=evals.sum()
        print(f"Layer {res['layer_idx']}:")
        for k in [32,64,128,256]:
            print(f"  top-{k} energy: {evals[:k].sum()/total:.4f}")
    del model; torch.cuda.empty_cache() if DEV=="cuda" else None

def step2():
    print("="*60+"\nGEMMA-2B: Utility evaluation\n"+"="*60)
    from two_channel.eval_utility import run_utility_grid
    n_layers=26
    mid=n_layers//2
    r,base=run_utility_grid(
        MODEL,layers=[mid,-1],k_values=[64,256],
        bits_values=[32,8],sigma_values=[0.0],
        device=DEV,prefix_len=32,n_eval=500,n_cal=300,
        subspace_dir=str(OUT/"subspace")
    )
    (OUT/"results").mkdir(parents=True,exist_ok=True)
    with open(OUT/"results/utility.json","w") as f:
        json.dump({"baseline_ppl":base,"results":r},f,indent=2)
    print(f"Baseline PPL: {base:.2f}, {len(r)} configs")

def step3():
    print("="*60+"\nGEMMA-2B: Leakage evaluation\n"+"="*60)
    from two_channel.eval_leakage import eval_leakage_grid
    n_layers=26
    mid=n_layers//2
    r=eval_leakage_grid(
        MODEL,layers=[mid,-1],k_values=[64,256],
        bits_values=[8],device=DEV,prefix_len=16,
        n_prefixes=30,batch_size=512,
        subspace_dir=str(OUT/"subspace")
    )
    with open(OUT/"results/leakage.json","w") as f:
        json.dump({"results":r},f,indent=2)

def step4():
    print("="*60+"\nGEMMA-2B: Plots\n"+"="*60)
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
    ax.set_title('Gemma-2B: Utility vs k (fp32)'); ax.legend(); ax.grid(alpha=0.3)

    ax=axes[1]
    for mode in ['behavior','identity','random']:
        pts=[(r['k'],r['margin_median']) for r in l['results']
             if r['bits']==8 and r['mode']==mode]
        if pts:
            pts.sort(); ks,m=zip(*pts)
            ax.plot(ks,m,color=COLORS[mode],marker='s',label=mode,linewidth=2)
    ax.set_xlabel('k'); ax.set_ylabel('Median margin')
    ax.set_title('Gemma-2B: Margin vs k (8-bit)'); ax.legend(); ax.grid(alpha=0.3)

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
    ax.set_title('Gemma-2B: Privacy-Utility Frontier'); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('Gemma-2B: Two-Channel Confirmation on Non-GPT Architecture',fontsize=14,fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/"plots/gemma2b_summary.png",dpi=150)
    plt.close(fig)

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

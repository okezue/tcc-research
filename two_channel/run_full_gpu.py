#!/usr/bin/env python3
import os, sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, json, time
from pathlib import Path

DEV="cuda" if torch.cuda.is_available() else "mps"
OUT=Path("artifacts")
OUT.mkdir(parents=True,exist_ok=True)

def step1_subspace():
    print("="*60+"\nSTEP 1: Computing behavior subspaces\n"+"="*60)
    from two_channel.compute_subspace import load_model, make_calibration_dataset, compute_gradient_covariance, save_subspace
    model,tok=load_model('openai-community/gpt2',DEV)
    ds=make_calibration_dataset(tok,n=5000,seq_len=33)
    for li in [6,-1]:
        res=compute_gradient_covariance(model,tok,li,ds,prefix_len=32,device=DEV,max_samples=5000)
        save_subspace(res,OUT/"subspace",'openai-community/gpt2',res["layer_idx"])
        evals=res["grad_eigenvalues"]
        total=evals.sum()
        print(f"Layer {res['layer_idx']}:")
        for k in [32,64,128,256]:
            print(f"  top-{k} energy: {evals[:k].sum()/total:.4f}")
    del model
    torch.cuda.empty_cache() if DEV=="cuda" else None

def step2_utility():
    print("="*60+"\nSTEP 2: Utility evaluation\n"+"="*60)
    from two_channel.eval_utility import run_utility_grid
    results,base=run_utility_grid(
        'openai-community/gpt2',
        layers=[6,-1],k_values=[32,64,128,256],
        bits_values=[32,16,8,6,4],sigma_values=[0.0],
        device=DEV,prefix_len=32,n_eval=2000,n_cal=1000,
        subspace_dir=str(OUT/"subspace")
    )
    (OUT/"results").mkdir(parents=True,exist_ok=True)
    with open(OUT/"results/utility.json","w") as f:
        json.dump({"baseline_ppl":base,"results":results},f,indent=2)
    print(f"Baseline PPL: {base:.4f}, {len(results)} configs")

def step3_leakage():
    print("="*60+"\nSTEP 3: Leakage / margin evaluation\n"+"="*60)
    from two_channel.eval_leakage import eval_leakage_grid
    results=eval_leakage_grid(
        'openai-community/gpt2',
        layers=[6,-1],k_values=[32,64,128,256],
        bits_values=[16,8,6,4],
        device=DEV,prefix_len=16,n_prefixes=100,batch_size=2048,
        subspace_dir=str(OUT/"subspace")
    )
    with open(OUT/"results/leakage.json","w") as f:
        json.dump({"results":results},f,indent=2)
    print(f"{len(results)} leakage configs done")

def step4_plots():
    print("="*60+"\nSTEP 4: Generating plots\n"+"="*60)
    from two_channel.plot_results import generate_all
    from two_channel.compute_subspace import get_layer_block, load_model
    model,_=load_model('openai-community/gpt2','cpu')
    layers=[]
    for li in [6,-1]:
        _,ai,_=get_layer_block(model,li)
        layers.append(ai)
    del model
    generate_all(
        str(OUT/"results/utility.json"),str(OUT/"results/leakage.json"),
        str(OUT/"subspace"),'openai-community/gpt2',layers,[32,64,128,256],
        str(OUT/"plots")
    )

if __name__=="__main__":
    t0=time.time()
    skip=set(sys.argv[1:])
    if "1" not in skip: step1_subspace()
    if "2" not in skip: step2_utility()
    if "3" not in skip: step3_leakage()
    if "4" not in skip: step4_plots()
    print(f"\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

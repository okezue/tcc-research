import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS={"behavior":"#e74c3c","identity":"#2ecc71","random":"#3498db","full":"#95a5a6"}
MARKERS={"behavior":"o","identity":"s","random":"^","full":"D"}

def load_results(utility_path,leakage_path):
    with open(utility_path) as f:
        u=json.load(f)
    with open(leakage_path) as f:
        l=json.load(f)
    return u,l

def plot1_dppl_vs_k(util_data,out_dir,layer=None):
    res=util_data["results"]
    base=util_data["baseline_ppl"]
    layers=sorted(set(r["layer"] for r in res))
    bits_all=sorted(set(r["bits"] for r in res if r["bits"]<32))

    for li in layers:
        if layer is not None and li!=layer:
            continue
        for b in bits_all:
            fig,ax=plt.subplots(figsize=(6,4))
            for mode in ["behavior","identity","random"]:
                pts=[(r["k"],r["dppl"]) for r in res
                     if r["layer"]==li and r["bits"]==b and r["mode"]==mode and r["sigma"]==0.0]
                if not pts:
                    continue
                pts.sort()
                ks,dppl=zip(*pts)
                ax.plot(ks,dppl,color=COLORS[mode],marker=MARKERS[mode],label=mode,linewidth=2)
            ax.set_xlabel("Subspace dim k")
            ax.set_ylabel("ΔPPL")
            ax.set_title(f"Layer {li}, {b}-bit quantization")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir/f"dppl_vs_k_layer{li}_b{b}.png",dpi=150)
            plt.close(fig)

def plot2_margin_vs_k(leak_data,out_dir,layer=None):
    res=leak_data["results"]
    layers=sorted(set(r["layer"] for r in res))
    bits_all=sorted(set(r["bits"] for r in res))

    for li in layers:
        if layer is not None and li!=layer:
            continue
        for b in bits_all:
            fig,ax=plt.subplots(figsize=(6,4))
            for mode in ["behavior","identity","random"]:
                pts=[(r["k"],r["margin_median"]) for r in res
                     if r["layer"]==li and r["bits"]==b and r["mode"]==mode]
                if not pts:
                    continue
                pts.sort()
                ks,m=zip(*pts)
                ax.plot(ks,m,color=COLORS[mode],marker=MARKERS[mode],label=mode,linewidth=2)
            ax.set_xlabel("Subspace dim k")
            ax.set_ylabel("Median margin")
            ax.set_title(f"Layer {li}, {b}-bit")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir/f"margin_vs_k_layer{li}_b{b}.png",dpi=150)
            plt.close(fig)

def plot3_unique_vs_k(leak_data,out_dir,layer=None):
    res=leak_data["results"]
    layers=sorted(set(r["layer"] for r in res))
    bits_all=sorted(set(r["bits"] for r in res))

    for li in layers:
        if layer is not None and li!=layer:
            continue
        for b in bits_all:
            fig,ax=plt.subplots(figsize=(6,4))
            for mode in ["behavior","identity","random"]:
                pts=[(r["k"],r["unique_frac"]) for r in res
                     if r["layer"]==li and r["bits"]==b and r["mode"]==mode]
                if not pts:
                    continue
                pts.sort()
                ks,uf=zip(*pts)
                ax.plot(ks,uf,color=COLORS[mode],marker=MARKERS[mode],label=mode,linewidth=2)
            ax.set_xlabel("Subspace dim k")
            ax.set_ylabel("Unique match rate")
            ax.set_title(f"Layer {li}, {b}-bit")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.05,1.05)
            fig.tight_layout()
            fig.savefig(out_dir/f"unique_vs_k_layer{li}_b{b}.png",dpi=150)
            plt.close(fig)

def plot4_privacy_utility_scatter(util_data,leak_data,out_dir):
    ur=util_data["results"]
    lr=leak_data["results"]

    fig,ax=plt.subplots(figsize=(7,5))
    for mode in ["behavior","identity","random"]:
        xs,ys=[],[]
        for u in ur:
            if u["mode"]!=mode or u["sigma"]!=0.0:
                continue
            match=[l for l in lr
                   if l["layer"]==u["layer"] and l["k"]==u["k"]
                   and l["mode"]==mode and l["bits"]==u["bits"]]
            if match:
                xs.append(u["dppl"])
                ys.append(match[0]["margin_median"])
        if xs:
            ax.scatter(xs,ys,color=COLORS[mode],marker=MARKERS[mode],label=mode,s=60,alpha=0.7)

    ax.set_xlabel("ΔPPL (utility cost)")
    ax.set_ylabel("Median margin (leakage robustness)")
    ax.set_title("Privacy-Utility Frontier")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"privacy_utility_frontier.png",dpi=150)
    plt.close(fig)

def table_energy(subspace_dir,model_id,layers,k_values):
    import torch
    rows=[]
    for li in layers:
        sp=Path(subspace_dir)/model_id.replace("/","_")/f"layer_{li}"
        evals=torch.load(sp/"grad_evals.pt",weights_only=True)
        total=evals.sum().item()
        for k in k_values:
            frac=evals[:k].sum().item()/total
            rows.append({"layer":li,"k":k,"energy_frac":frac})
    return rows

def generate_all(utility_path,leakage_path,subspace_dir,model_id,layers,k_values,out_dir):
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)

    u,l=load_results(utility_path,leakage_path)
    plot1_dppl_vs_k(u,out_dir)
    plot2_margin_vs_k(l,out_dir)
    plot3_unique_vs_k(l,out_dir)
    plot4_privacy_utility_scatter(u,l,out_dir)

    energy=table_energy(subspace_dir,model_id,layers,k_values)
    with open(out_dir/"energy_table.json","w") as f:
        json.dump(energy,f,indent=2)

    print(f"All plots saved to {out_dir}")
    print("\nGradient energy captured by top-k:")
    for r in energy:
        print(f"  Layer {r['layer']}, k={r['k']}: {r['energy_frac']:.4f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--utility",default="artifacts/results/utility.json")
    p.add_argument("--leakage",default="artifacts/results/leakage.json")
    p.add_argument("--subspace-dir",default="artifacts/subspace")
    p.add_argument("--model-id",default="openai-community/gpt2")
    p.add_argument("--layers",nargs="+",type=int,default=[6,11])
    p.add_argument("--k-values",nargs="+",type=int,default=[32,64,128,256])
    p.add_argument("--out-dir",default="artifacts/plots")
    args=p.parse_args()

    generate_all(
        args.utility,args.leakage,args.subspace_dir,
        args.model_id,args.layers,args.k_values,args.out_dir
    )

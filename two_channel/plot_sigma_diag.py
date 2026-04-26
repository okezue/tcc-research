import argparse,json,os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--summary",default="artifacts/sigma_diag_validate/summary.json")
    ap.add_argument("--out",default="paper_figs/fig_sigma_diag_alpha.pdf")
    a=ap.parse_args()
    s=json.load(open(a.summary))["summary"]
    by_mdl={}
    for r in s:
        m=r["model"].split("/")[-1]
        by_mdl.setdefault(m,[]).append(r)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharex=True)
    plt.rcParams.update({'font.family':'serif','pdf.fonttype':42})
    colors={'gpt2':'#4a8ab5','Mistral-7B-v0.1':'#c96a47','phi-2':'#8a5a8a','Qwen3-14B':'#5a9a5a','DeepSeek-R1-Distill-Qwen-14B':'#aa4a4a'}
    short={'gpt2':'GPT-2','Mistral-7B-v0.1':'Mistral-7B','phi-2':'Phi-2','Qwen3-14B':'Qwen3-14B','DeepSeek-R1-Distill-Qwen-14B':'DeepSeek-R1-14B'}
    for mdl,recs in sorted(by_mdl.items()):
        col=colors.get(mdl,'gray')
        nm=short.get(mdl,mdl)
        all_a=sorted([float(a) for a in recs[0]["worst_curve"]])
        Ws=np.array([[r["worst_curve"][f"{a:.2f}"][0] for a in all_a] for r in recs])
        Ts=np.array([[r["top1_curve"][f"{a:.2f}"][0] for a in all_a] for r in recs])
        Wn=Ws/Ws[:,0:1]
        m_w=Wn.mean(axis=0)
        s_w=Wn.std(axis=0)
        m_t=Ts.mean(axis=0)
        s_t=Ts.std(axis=0)
        axes[0].plot(all_a,m_w,'-o',color=col,label=nm,markersize=4,lw=1.2)
        axes[0].fill_between(all_a,m_w-s_w,m_w+s_w,color=col,alpha=0.15)
        axes[1].plot(all_a,m_t,'-o',color=col,label=nm,markersize=4,lw=1.2)
        axes[1].fill_between(all_a,m_t-s_t,m_t+s_t,color=col,alpha=0.15)
    axes[0].axvline(1.0,color='k',lw=0.5,ls=':',alpha=0.5)
    axes[1].axvline(1.0,color='k',lw=0.5,ls=':',alpha=0.5)
    axes[0].set_xlabel(r'$\alpha$ (Fisher exponent in $\Sigma_\alpha \propto \mathrm{diag}(F)^{-\alpha}$)')
    axes[0].set_ylabel(r'worst-case Mahalanobis (normalised to $\alpha=0$)')
    axes[0].set_title('Theorem-aligned scalar')
    axes[1].set_xlabel(r'$\alpha$ (Fisher exponent)')
    axes[1].set_ylabel('retrieval attack top-1')
    axes[1].set_title('Empirical attack')
    axes[0].legend(fontsize=8,loc='upper right')
    axes[0].spines['top'].set_visible(False);axes[0].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False);axes[1].spines['right'].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out)or".",exist_ok=True)
    plt.savefig(a.out,bbox_inches="tight",dpi=200)
    print(f"wrote {a.out}")

if __name__=="__main__":
    main()

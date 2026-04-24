#!/usr/bin/env python3
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import json,time,gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/inversion_extended")
OUT.mkdir(parents=True,exist_ok=True)

MODEL="openai-community/gpt2"
LAYER=11
D_MODEL=768

def make_ds(tok,n=200,sl=64,seed=42):
    from datasets import load_dataset
    ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    torch.manual_seed(seed)
    out=[]
    for row in ds:
        txt=row["text"].strip()
        if len(txt)<80: continue
        ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=sl:
            s=torch.randint(0,len(ids)-sl+1,(1,)).item()
            out.append(torch.tensor(ids[s:s+sl],dtype=torch.long))
            if len(out)>=n: break
    return out

def load_subspace(k):
    sp=Path(f"artifacts/subspace/openai-community_gpt2/layer_{LAYER}")
    if not sp.exists():
        return None
    evecs=torch.load(sp/"grad_evecs.pt",weights_only=True)
    return evecs[:,:k].to(torch.float32)

def project(h,U):
    return h@U@U.T

def project_complement(h,U):
    return h-project(h,U)

def get_final_hidden(model,ids,dev):
    with torch.no_grad():
        o=model(ids.unsqueeze(0).to(dev),output_hidden_states=True)
    return o.hidden_states[-1][0,-1,:]

def margin_from_final_state(model,ds,U,mode,dev,pl,n_eval=100,n_distractors=50):
    valid=[s for s in ds if len(s)>=pl][:n_eval]
    tok_margins=[]
    tok_ranks=[]
    em=0
    for seq in tqdm(valid,desc=f"{mode} pl={pl}"):
        prefix=seq[:pl].to(dev)
        h_true=get_final_hidden(model,prefix,dev)
        if mode=="behavior" and U is not None:
            h_true=project(h_true.unsqueeze(0),U).squeeze(0)
        elif mode=="identity" and U is not None:
            h_true=project_complement(h_true.unsqueeze(0),U).squeeze(0)
        all_correct=True
        for t in range(pl):
            true_tok=prefix[t].item()
            cands=[true_tok]
            torch.manual_seed(t*1000+hash(tuple(prefix.cpu().tolist()))%10000)
            V=model.config.vocab_size
            rnd=torch.randint(0,V,(n_distractors*2,))
            for r in rnd:
                if r.item()!=true_tok and len(cands)<=n_distractors:
                    cands.append(r.item())
            dists=[]
            for c in cands:
                mut=prefix.clone()
                mut[t]=c
                h_c=get_final_hidden(model,mut,dev)
                if mode=="behavior" and U is not None:
                    h_c=project(h_c.unsqueeze(0),U).squeeze(0)
                elif mode=="identity" and U is not None:
                    h_c=project_complement(h_c.unsqueeze(0),U).squeeze(0)
                dists.append(torch.norm(h_true-h_c).item())
            d_true=dists[0]
            d_others=dists[1:]
            rank=sum(1 for d in d_others if d<=d_true)+1
            margin=min(d_others)-d_true if d_others else 0
            tok_margins.append(margin)
            tok_ranks.append(rank)
            if rank!=1:
                all_correct=False
        if all_correct:
            em+=1
    return {
        "exact_match":em/max(len(valid),1),
        "margin_median":float(np.median(tok_margins)),
        "margin_mean":float(np.mean(tok_margins)),
        "margin_p10":float(np.percentile(tok_margins,10)),
        "mean_rank":float(np.mean(tok_ranks)),
        "median_rank":float(np.median(tok_ranks)),
        "rank1_frac":sum(1 for r in tok_ranks if r==1)/max(len(tok_ranks),1),
    }

def nn_recovery(model,ds,U,mode,dev,pl,n_eval=200):
    valid=[s for s in ds if len(s)>=pl]
    pool=valid[:n_eval]
    hs=[]
    for seq in tqdm(pool,desc=f"embed {mode} pl={pl}"):
        prefix=seq[:pl].to(dev)
        h=get_final_hidden(model,prefix,dev)
        if mode=="behavior" and U is not None:
            h=project(h.unsqueeze(0),U).squeeze(0)
        elif mode=="identity" and U is not None:
            h=project_complement(h.unsqueeze(0),U).squeeze(0)
        hs.append(h.cpu())
    H=torch.stack(hs)
    D=torch.cdist(H.unsqueeze(0),H.unsqueeze(0)).squeeze(0)
    D.fill_diagonal_(float('inf'))
    nn_idx=D.argmin(dim=1)
    tok_match=[]
    for i in range(len(pool)):
        j=nn_idx[i].item()
        p_i=pool[i][:pl]
        p_j=pool[j][:pl]
        m=(p_i==p_j).float().mean().item()
        tok_match.append(m)
    margins=[]
    for i in range(len(pool)):
        row=D[i]
        sorted_d=row.sort().values
        margins.append((sorted_d[1]-sorted_d[0]).item())
    return {
        "nn_tok_overlap_mean":float(np.mean(tok_match)),
        "nn_tok_overlap_median":float(np.median(tok_match)),
        "nn_margin_median":float(np.median(margins)),
        "nn_margin_mean":float(np.mean(margins)),
    }

def main():
    t0=time.time()
    print("="*60+"\nInversion Experiment v2: Final-Position Recovery\n"+"="*60)
    from transformers import AutoModelForCausalLM,AutoTokenizer
    model=AutoModelForCausalLM.from_pretrained(MODEL,output_hidden_states=True)
    model.eval().to(DEV)
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    ds=make_ds(tok,n=300,sl=64)

    prompt_lens=[4,8,16,32]
    k_vals=[64,128,256]
    n_margin=50
    n_nn=200

    subspaces={}
    for k in k_vals:
        U=load_subspace(k)
        if U is not None:
            subspaces[k]=U.to(DEV)
            print(f"Loaded subspace k={k}: {U.shape}")
        else:
            print(f"Subspace not found for k={k}, will use random")
            torch.manual_seed(42)
            M=torch.randn(D_MODEL,k)
            Q,_=torch.linalg.qr(M)
            subspaces[k]=Q.to(DEV)

    results=[]
    configs=[]
    for pl in prompt_lens:
        configs.append(("full",pl,None,None))
        for k in k_vals:
            configs.append(("behavior",pl,k,subspaces[k]))
            configs.append(("identity",pl,k,subspaces[k]))

    for mode,pl,k,U in configs:
        label=f"mode={mode} pl={pl} k={k or 'full'}"
        print(f"\n--- {label} ---")
        mr=margin_from_final_state(model,ds,U,mode,DEV,pl,n_eval=n_margin,n_distractors=50)
        nr=nn_recovery(model,ds,U,mode,DEV,pl,n_eval=n_nn)
        entry={"mode":mode,"prompt_length":pl,"k":k if k else D_MODEL}
        entry.update(mr)
        entry.update(nr)
        results.append(entry)
        print(f"  margin_med={mr['margin_median']:.3f} rank1={mr['rank1_frac']:.2%} em={mr['exact_match']:.2%}")
        print(f"  nn_overlap={nr['nn_tok_overlap_mean']:.3f} nn_margin={nr['nn_margin_median']:.3f}")
        gc.collect()
        if DEV=="cuda": torch.cuda.empty_cache()

    with open(OUT/"inversion_v2.json","w") as f:
        json.dump({"results":results,"model":MODEL,"layer":LAYER,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")},f,indent=2)

    fig,axes=plt.subplots(2,3,figsize=(18,10))
    modes_plot=["full"]+[f"behavior_k{k}" for k in k_vals]+[f"identity_k{k}" for k in k_vals]
    beh_margins={pl:[] for pl in prompt_lens}
    id_margins={pl:[] for pl in prompt_lens}
    full_margins={pl:[] for pl in prompt_lens}
    for r in results:
        pl=r["prompt_length"]
        if r["mode"]=="full":
            full_margins[pl].append(r["margin_median"])
        elif r["mode"]=="behavior":
            beh_margins[pl].append((r["k"],r["margin_median"]))
        elif r["mode"]=="identity":
            id_margins[pl].append((r["k"],r["margin_median"]))

    ax=axes[0,0]
    for pl in prompt_lens:
        fm=full_margins[pl][0] if full_margins[pl] else 0
        bm=[v for _,v in sorted(beh_margins[pl])]
        im=[v for _,v in sorted(id_margins[pl])]
        x=list(range(len(k_vals)))
        ax.plot(x,bm,'o-',label=f'beh pl={pl}')
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in sorted(k_vals)])
    ax.set_xlabel("k")
    ax.set_ylabel("Margin (median)")
    ax.set_title("Behavior Subspace Margins")
    ax.legend(fontsize=7)

    ax=axes[0,1]
    for pl in prompt_lens:
        im=[v for _,v in sorted(id_margins[pl])]
        ax.plot(list(range(len(k_vals))),im,'s-',label=f'id pl={pl}')
    ax.set_xticks(list(range(len(k_vals))))
    ax.set_xticklabels([str(k) for k in sorted(k_vals)])
    ax.set_xlabel("k")
    ax.set_ylabel("Margin (median)")
    ax.set_title("Identity Complement Margins")
    ax.legend(fontsize=7)

    ax=axes[0,2]
    for pl in prompt_lens:
        beh_r1=[r["rank1_frac"] for r in results if r["mode"]=="behavior" and r["prompt_length"]==pl]
        id_r1=[r["rank1_frac"] for r in results if r["mode"]=="identity" and r["prompt_length"]==pl]
        x_pos=prompt_lens.index(pl)
        ax.bar(x_pos-0.15,np.mean(beh_r1),0.3,label='behavior' if pl==prompt_lens[0] else '',color='tab:blue',alpha=0.7)
        ax.bar(x_pos+0.15,np.mean(id_r1),0.3,label='identity' if pl==prompt_lens[0] else '',color='tab:orange',alpha=0.7)
    ax.set_xticks(range(len(prompt_lens)))
    ax.set_xticklabels([str(pl) for pl in prompt_lens])
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Rank-1 fraction")
    ax.set_title("Token Recovery Accuracy")
    ax.legend()

    ax=axes[1,0]
    for r in results:
        if r["mode"]=="behavior":
            ax.scatter(r["k"],r["nn_tok_overlap_mean"],c='tab:blue',alpha=0.5,s=20+r["prompt_length"]*3)
        elif r["mode"]=="identity":
            ax.scatter(r["k"],r["nn_tok_overlap_mean"],c='tab:orange',alpha=0.5,s=20+r["prompt_length"]*3)
    ax.set_xlabel("k")
    ax.set_ylabel("NN Token Overlap")
    ax.set_title("Nearest-Neighbor Overlap")

    ax=axes[1,1]
    for pl in prompt_lens:
        beh_em=[r["exact_match"] for r in results if r["mode"]=="behavior" and r["prompt_length"]==pl]
        id_em=[r["exact_match"] for r in results if r["mode"]=="identity" and r["prompt_length"]==pl]
        full_em=[r["exact_match"] for r in results if r["mode"]=="full" and r["prompt_length"]==pl]
        x_pos=prompt_lens.index(pl)
        ax.bar(x_pos-0.2,np.mean(beh_em) if beh_em else 0,0.2,color='tab:blue',alpha=0.7)
        ax.bar(x_pos,np.mean(id_em) if id_em else 0,0.2,color='tab:orange',alpha=0.7)
        ax.bar(x_pos+0.2,np.mean(full_em) if full_em else 0,0.2,color='tab:green',alpha=0.7)
    ax.set_xticks(range(len(prompt_lens)))
    ax.set_xticklabels([str(pl) for pl in prompt_lens])
    ax.set_xlabel("Prompt length")
    ax.set_ylabel("Exact Match Rate")
    ax.set_title("Exact Prefix Recovery (blue=beh, orange=id, green=full)")

    ax=axes[1,2]
    summary=[]
    for r in results:
        summary.append(f"{r['mode'][:3]} k={r['k']:>4} pl={r['prompt_length']:>2}: "
                       f"r1={r['rank1_frac']:.0%} mar={r['margin_median']:.1f}")
    ax.axis('off')
    ax.text(0.02,0.98,"\n".join(summary),transform=ax.transAxes,fontsize=6,
            verticalalignment='top',fontfamily='monospace')
    ax.set_title("Results Summary")

    fig.suptitle("Inversion v2: Final-Position Hidden State Recovery\n(GPT-2 Small, Layer 11)",fontsize=13,fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/"inversion_v2.png",dpi=150)
    fig.savefig(OUT/"inversion_v2.pdf")
    plt.close(fig)
    print(f"\nResults saved to {OUT}")
    print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

if __name__=="__main__":
    main()

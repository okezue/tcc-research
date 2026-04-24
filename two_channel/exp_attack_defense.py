#!/usr/bin/env python3
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import json,time,gc,argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/attack_defense")
OUT.mkdir(parents=True,exist_ok=True)

def make_ds(tok,n=5000,sl=33,seed=42):
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

def get_subspace(model,tok,ds,layer,dev,k=128,n_cal=2000):
    from two_channel.compute_subspace import get_layer_block
    blk,li,nl=get_layer_block(model,layer)
    d=model.config.hidden_size
    V=model.config.vocab_size
    G=torch.zeros(d,d,dtype=torch.float64)
    cnt=0
    for seq in tqdm(ds[:n_cal],desc="grad cov"):
        x=seq[:-1].unsqueeze(0).to(dev)
        y=seq[-1].to(dev)
        cap=[None]
        def hk(m,i,o,c=cap):
            oo=o[0] if isinstance(o,tuple) else o
            c[0]=oo
        h=blk.register_forward_hook(hk)
        out=model(x)
        h.remove()
        hl=cap[0][0,-1,:].detach().requires_grad_(True)
        logits=model.lm_head(model.transformer.ln_f(hl) if hasattr(model,'transformer') else model.model.norm(hl))
        loss=F.cross_entropy(logits.unsqueeze(0),y.unsqueeze(0))
        g=torch.autograd.grad(loss,hl)[0].detach().float().cpu()
        G+=torch.outer(g.to(torch.float64),g.to(torch.float64))
        cnt+=1
    G/=cnt
    evals,evecs=torch.linalg.eigh(G.float())
    idx=evals.argsort(descending=True)
    evecs=evecs[:,idx]
    evals=evals[idx]
    U=evecs[:,:k].to(torch.float32).to(dev)
    efrac=evals[:k].sum()/evals.sum()
    r95=0
    cs=evals.cumsum(0)/evals.sum()
    for i,c in enumerate(cs):
        if c>=0.95:
            r95=i+1; break
    return U,evals,efrac.item(),r95

def embed_pool(model,pool,layer,dev,U_B=None,mode="full",ctx=32):
    from two_channel.compute_subspace import get_layer_block
    blk,li,_=get_layer_block(model,layer)
    hs=[]
    cap=[None]
    def hk(m,i,o,c=cap):
        oo=o[0] if isinstance(o,tuple) else o
        c[0]=oo.detach()
    handle=blk.register_forward_hook(hk)
    for seq in tqdm(pool,desc=f"embed {mode}",leave=False):
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad():
            model(x)
        h=cap[0][0,-1,:].float()
        if mode=="behavior" and U_B is not None:
            h=h@U_B@U_B.T
        elif mode=="identity" and U_B is not None:
            h=h-h@U_B@U_B.T
        hs.append(h.cpu())
    handle.remove()
    return torch.stack(hs)

def alpha_attack(H_q,H_bank,U_B,alpha,dev):
    d=H_q.shape[1]
    U=U_B.cpu()
    H_q_B=H_q@U@U.T
    H_q_I=H_q-H_q_B
    H_b_B=H_bank@U@U.T
    H_b_I=H_bank-H_b_B
    D_I=torch.cdist(H_q_I,H_b_I)
    D_B=torch.cdist(H_q_B,H_b_B)
    D=D_I**2+alpha*D_B**2
    return D

def eval_retrieval(D,query_idx):
    n=D.shape[0]
    top1=0; mrr=0; r10=0; ranks=[]
    for i in range(n):
        row=D[i]
        rank=(row<row[query_idx[i]]).sum().item()+1
        ranks.append(rank)
        if rank==1: top1+=1
        if rank<=10: r10+=1
        mrr+=1.0/rank
    return {
        "top1":top1/n,"mrr":mrr/n,"recall10":r10/n,
        "median_rank":float(np.median(ranks)),
        "mean_rank":float(np.mean(ranks)),
    }

def build_behavior_hard_pool(model,tok,bank,layer,dev,U_B,n_query=1000,n_hard=100,ctx=32):
    from two_channel.compute_subspace import get_layer_block
    blk,li,_=get_layer_block(model,layer)
    logits_all=[]
    cap=[None]
    def hk(m,i,o,c=cap):
        oo=o[0] if isinstance(o,tuple) else o
        c[0]=oo.detach()
    handle=blk.register_forward_hook(hk)
    for seq in tqdm(bank[:n_query+5000],desc="behavior logits",leave=False):
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad():
            out=model(x)
        lp=F.log_softmax(out.logits[0,-1,:],dim=-1)
        logits_all.append(lp.cpu())
    handle.remove()
    L=torch.stack(logits_all)
    return L

def add_noise_complement(H,U_B,sigma):
    U=U_B.cpu()
    d=H.shape[1]
    k=U.shape[1]
    H_B=H@U@U.T
    H_I=H-H_B
    noise=torch.randn_like(H_I)*sigma
    noise_I=noise-noise@U@U.T
    return H_B+H_I+noise_I

def add_noise_isotropic(H,sigma):
    return H+torch.randn_like(H)*sigma

def add_noise_behavior(H,U_B,sigma):
    U=U_B.cpu()
    noise=torch.randn_like(H)*sigma
    noise_B=noise@U@U.T
    return H+noise_B

def project_behavior_plus_noise(H,U_B,sigma):
    U=U_B.cpu()
    H_B=H@U@U.T
    noise=torch.randn_like(H_B)*sigma
    noise_I=noise-noise@U@U.T
    return H_B+noise_I

def project_behavior_dp(H,U_B,sigma):
    U=U_B.cpu()
    H_B=H@U@U.T
    k=U.shape[1]
    noise_B=torch.randn(H.shape[0],k)*sigma
    return H_B+noise_B@U.T

def measure_kl(model,ds,layer,dev,noise_fn,U_B,sigma,n_eval=500,ctx=32):
    from two_channel.compute_subspace import get_layer_block
    blk,li,_=get_layer_block(model,layer)
    kls=[]
    top1s=[]
    for seq in ds[:n_eval]:
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad():
            out_clean=model(x)
            bl=out_clean.logits[0,-1,:]
            bp=F.softmax(bl,dim=-1)
            t1_clean=bl.argmax().item()
        noisy_h=[None]
        cap=[None]
        def cap_hk(m,i,o,c=cap):
            oo=o[0] if isinstance(o,tuple) else o
            c[0]=oo.detach()
        h1=blk.register_forward_hook(cap_hk)
        with torch.no_grad(): model(x)
        h1.remove()
        h_clean=cap[0][0,-1,:].float()
        h_noisy=noise_fn(h_clean.unsqueeze(0).cpu(),U_B,sigma).to(dev).squeeze(0)
        noisy_h[0]=h_noisy
        def inject_hk(m,i,o,nh=noisy_h):
            oo=o[0] if isinstance(o,tuple) else o
            oo=oo.clone()
            oo[0,-1,:]=nh[0].to(oo.dtype)
            if isinstance(o,tuple): return (oo,)+o[1:]
            return oo
        h2=blk.register_forward_hook(inject_hk)
        with torch.no_grad():
            out_noisy=model(x)
        h2.remove()
        logits_n=out_noisy.logits[0,-1,:]
        kl=F.kl_div(F.log_softmax(logits_n,dim=-1),bp,reduction='sum',log_target=False).item()
        t1_noisy=logits_n.argmax().item()
        kls.append(kl)
        top1s.append(int(t1_clean==t1_noisy))
    return float(np.mean(kls)),float(np.mean(top1s))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=None)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    mname=args.model
    dev=args.device
    t0=time.time()

    print("="*60+f"\nAttack-Defense Experiment: {mname}\n"+"="*60)

    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(mname)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    is_gpt2="gpt2" in mname.lower()
    if is_gpt2:
        model=AutoModelForCausalLM.from_pretrained(mname,output_hidden_states=True)
    else:
        model=AutoModelForCausalLM.from_pretrained(mname,torch_dtype=torch.bfloat16,output_hidden_states=True)
    model.eval().to(dev)
    for pp in model.parameters(): pp.requires_grad_(False)

    d=model.config.hidden_size
    nl=model.config.num_hidden_layers
    layer=args.layer if args.layer else nl//2
    k=args.k
    ctx=32
    print(f"d={d} layers={nl} layer={layer} k={k} ctx={ctx}")

    ds=make_ds(tok,n=8000,sl=ctx+1)
    calib=ds[:2000]
    bank=ds[2000:7000]
    val_q=ds[7000:7500]
    test_q=ds[7500:8000]
    print(f"Data: calib={len(calib)} bank={len(bank)} val={len(val_q)} test={len(test_q)}")

    print("\n--- Computing Fisher basis ---")
    U_B,evals,efrac,r95=get_subspace(model,tok,calib,layer,dev,k=k,n_cal=min(2000,len(calib)))
    print(f"Energy in top-{k}: {efrac:.3f}, r_95={r95}")

    print("\n--- Embedding bank + queries ---")
    H_bank=embed_pool(model,bank,layer,dev,ctx=ctx)
    H_val=embed_pool(model,val_q,layer,dev,ctx=ctx)
    H_test=embed_pool(model,test_q,layer,dev,ctx=ctx)
    gc.collect()
    if dev=="cuda": torch.cuda.empty_cache()

    N_bank=len(bank)
    print(f"\nRetrieval setup: bank={N_bank}, query=subset of bank (clean vs noised)")
    query_ids=list(range(0,min(500,N_bank),1))
    H_queries=H_bank[query_ids]

    print("\n--- ATTACK: α-sweep (clean queries vs clean bank) ---")
    alphas=[0.0,0.01,0.1,0.3,0.5,1.0]
    attack_results=[]
    for alpha in alphas:
        D=alpha_attack(H_queries,H_bank,U_B,alpha,dev)
        for i,qi in enumerate(query_ids):
            D[i,qi]=float('inf')
        nn_idx=D.argmin(dim=1)
        n_q=len(query_ids)
        top1=sum(1 for i,qi in enumerate(query_ids) if nn_idx[i].item()==qi)/n_q
        ranks=[]
        for i,qi in enumerate(query_ids):
            D[i,qi]=0
            row=D[i]
            rank=(row<row[qi]).sum().item()+1
            ranks.append(rank)
            D[i,qi]=float('inf')
        mrr=sum(1.0/r for r in ranks)/n_q
        r10=sum(1 for r in ranks if r<=10)/n_q
        r={"top1":top1,"mrr":mrr,"recall10":r10,"median_rank":float(np.median(ranks)),"alpha":alpha}
        attack_results.append(r)
        print(f"  α={alpha:.2f}: top1={r['top1']:.3f} mrr={r['mrr']:.3f} r@10={r['recall10']:.3f} med_rank={r['median_rank']:.0f}")

    best_alpha=max(attack_results,key=lambda x:x["mrr"])["alpha"]
    print(f"  Best α={best_alpha:.2f}")

    test_attack={"alpha_sweep":attack_results,"best_alpha":best_alpha}

    print("\n--- DEFENSE: Pareto frontier ---")
    sigmas=[0.0,0.1,0.3,0.5,1.0,2.0,5.0,10.0]
    mechanisms={
        "complement_noise":lambda H,U,s: add_noise_complement(H,U,s),
        "isotropic":lambda H,U,s: add_noise_isotropic(H,s),
        "behavior_noise":lambda H,U,s: add_noise_behavior(H,U,s),
        "proj_replace":lambda H,U,s: project_behavior_plus_noise(H,U,s),
        "proj_dp":lambda H,U,s: project_behavior_dp(H,U,s),
    }
    defense_results=[]
    for mech_name,mech_fn in mechanisms.items():
        print(f"\n  Mechanism: {mech_name}")
        for sigma in sigmas:
            H_noisy=mech_fn(H_queries,U_B,sigma)
            for a_test in [best_alpha,0.0,1.0]:
                D=alpha_attack(H_noisy,H_bank,U_B,a_test,dev)
                n_q=len(query_ids)
                ranks=[]
                for i,qi in enumerate(query_ids):
                    row=D[i]
                    rank=(row<row[qi]).sum().item()+1
                    ranks.append(rank)
                atk_top1=sum(1 for r in ranks if r==1)/n_q
                atk_mrr=sum(1.0/r for r in ranks)/n_q
                atk_r10=sum(1 for r in ranks if r<=10)/n_q
                atk_med=float(np.median(ranks))
                mean_kl,top1_agree=0.0,1.0
                if sigma>0 and a_test==best_alpha:
                    mean_kl,top1_agree=measure_kl(model,bank,layer,dev,
                        lambda h,u,s,fn=mech_fn:fn(h,u,s),U_B,sigma,n_eval=200,ctx=ctx)
                entry={"mechanism":mech_name,"sigma":sigma,"alpha":a_test,
                       "mean_kl":mean_kl,"top1_agree":top1_agree,
                       "attack_top1":atk_top1,"attack_mrr":atk_mrr,
                       "attack_r10":atk_r10,"attack_med_rank":atk_med}
                defense_results.append(entry)
                if a_test==best_alpha:
                    print(f"    σ={sigma:.1f}: kl={mean_kl:.4f} t1_agree={top1_agree:.3f} atk_top1={atk_top1:.3f} atk_mrr={atk_mrr:.3f} med_rank={atk_med:.0f}")
        gc.collect()
        if dev=="cuda": torch.cuda.empty_cache()

    results={
        "model":mname,"layer":layer,"k":k,"d":d,"ctx":ctx,
        "r95":r95,"energy_frac":efrac,
        "attack_val":attack_results,
        "attack_test":test_attack,
        "best_alpha":best_alpha,
        "defense":defense_results,
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s":time.time()-t0,
    }
    mslug=mname.replace("/","_")
    with open(OUT/f"attack_defense_{mslug}.json","w") as f:
        json.dump(results,f,indent=2)

    fig,axes=plt.subplots(1,2,figsize=(14,6))
    ax=axes[0]
    for mech_name in mechanisms:
        pts=[(r["mean_kl"],r["attack_top1"]) for r in defense_results if r["mechanism"]==mech_name and r["sigma"]>0]
        if pts:
            pts.sort()
            ax.plot([p[0] for p in pts],[p[1] for p in pts],'o-',label=mech_name,markersize=4)
    ax.set_xlabel("Mean KL (utility cost)")
    ax.set_ylabel("Attack Top-1 (privacy leak)")
    ax.set_title("Defense Pareto Frontier")
    ax.legend(fontsize=7)
    ax.set_xscale('log')

    ax=axes[1]
    for r in attack_results:
        ax.bar(f"α={r['alpha']:.2f}",r["top1"],alpha=0.7)
    ax.set_ylabel("Top-1 Retrieval")
    ax.set_title("Attack: α-sweep (val)")

    fig.suptitle(f"Attack & Defense: {mname}, layer {layer}, k={k}",fontsize=13,fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT/f"attack_defense_{mslug}.png",dpi=150)
    fig.savefig(OUT/f"attack_defense_{mslug}.pdf")
    plt.close(fig)
    print(f"\nResults saved to {OUT}")
    print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import json,time,gc,argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/optimal_defense")
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

def get_layer_block(model,layer):
    if hasattr(model,'transformer') and hasattr(model.transformer,'h'):
        return model.transformer.h[layer],layer,len(model.transformer.h)
    if hasattr(model,'model') and hasattr(model.model,'layers'):
        return model.model.layers[layer],layer,len(model.model.layers)
    raise ValueError("cannot find layers")

def compute_fisher_avg(model,ds,layer,dev,n_cal=2000,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    d=model.config.hidden_size
    F_avg=torch.zeros(d,d,dtype=torch.float64)
    cnt=0
    for seq in tqdm(ds[:n_cal],desc="Fisher"):
        x=seq[:ctx].unsqueeze(0).to(dev)
        cap=[None]
        def hkc(m,i,o,c=cap):
            c[0]=(o[0] if isinstance(o,tuple) else o).detach()
        h1=blk.register_forward_hook(hkc)
        with torch.no_grad(): model(x)
        h1.remove()
        h_orig=cap[0][0,-1,:].float()
        h_var=h_orig.clone().requires_grad_(True)
        def inject(m,i,o,hv=h_var):
            oo=o[0] if isinstance(o,tuple) else o
            oo=oo.clone()
            oo[0,-1,:]=hv.to(oo.dtype)
            if isinstance(o,tuple): return (oo,)+o[1:]
            return oo
        h2=blk.register_forward_hook(inject)
        out=model(x)
        h2.remove()
        logits=out.logits[0,-1,:].float()
        p=F.softmax(logits,dim=-1).detach()
        lp=F.log_softmax(logits,dim=-1)
        Fh=torch.zeros(d,d,dtype=torch.float64)
        top_k_v=torch.topk(p,10).indices.tolist()
        for v in top_k_v:
            g=torch.autograd.grad(lp[v],h_var,retain_graph=True)[0].detach().float().cpu()
            pv=p[v].item()
            gg=g.to(torch.float64)
            Fh+=pv*torch.outer(gg,gg)
        F_avg+=Fh
        cnt+=1
    F_avg/=cnt
    return F_avg.float()

def compute_id_cov(H_bank,n_pairs=5000):
    N=H_bank.shape[0]
    d=H_bank.shape[1]
    S=torch.zeros(d,d,dtype=torch.float64)
    torch.manual_seed(42)
    idx1=torch.randint(0,N,(n_pairs,))
    for i in idx1:
        row=torch.cdist(H_bank[i:i+1],H_bank).squeeze(0)
        row[i]=float('inf')
        j=row.argmin()
        delta=(H_bank[i]-H_bank[j]).double()
        S+=torch.outer(delta,delta)
    S/=n_pairs
    return S.float()

def embed_bank(model,ds,layer,dev,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    hs=[]
    cap=[None]
    def hk(m,i,o,c=cap):
        c[0]=o[0].detach() if isinstance(o,tuple) else o.detach()
    h=blk.register_forward_hook(hk)
    for seq in tqdm(ds,desc="embed",leave=False):
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad(): model(x)
        hs.append(cap[0][0,-1,:].float().cpu())
    h.remove()
    return torch.stack(hs)

def gen_eigendecomp(S,F,reg_frac=0.01):
    d=S.shape[0]
    F_sym=(F+F.T)/2
    F_evals,F_evecs=torch.linalg.eigh(F_sym)
    F_max=F_evals.max().item()
    F_floor=reg_frac*F_max
    F_clamped=F_evals.clamp(min=F_floor)
    F_reg=F_evecs@torch.diag(F_clamped)@F_evecs.T
    L=torch.linalg.cholesky(F_reg)
    Linv=torch.linalg.solve_triangular(L,torch.eye(d),upper=False)
    M=Linv@S@Linv.T
    M=(M+M.T)/2
    evals,evecs=torch.linalg.eigh(M)
    idx=evals.argsort(descending=True)
    evals=evals[idx]
    evecs=evecs[:,idx]
    V=Linv.T@evecs
    return evals,V,F_clamped,F_evecs

def measure_kl(model,ds,layer,dev,noise_fn,n_eval=200,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    kls=[];t1s=[]
    for seq in ds[:n_eval]:
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad():
            out=model(x)
            bp=F.softmax(out.logits[0,-1,:].float(),dim=-1)
            t1_clean=out.logits[0,-1,:].argmax().item()
        cap=[None]
        def chk(m,i,o,c=cap):
            c[0]=o[0].detach() if isinstance(o,tuple) else o.detach()
        h1=blk.register_forward_hook(chk)
        with torch.no_grad(): model(x)
        h1.remove()
        h_clean=cap[0][0,-1,:].float()
        h_noisy=noise_fn(h_clean.cpu()).to(dev)
        def inject(m,i,o,nh=h_noisy):
            oo=o[0] if isinstance(o,tuple) else o
            oo=oo.clone()
            oo[0,-1,:]=nh.to(oo.dtype)
            if isinstance(o,tuple): return (oo,)+o[1:]
            return oo
        h2=blk.register_forward_hook(inject)
        with torch.no_grad(): out2=model(x)
        h2.remove()
        lp=F.log_softmax(out2.logits[0,-1,:].float(),dim=-1)
        kls.append(F.kl_div(lp,bp,reduction='sum',log_target=False).item())
        t1s.append(int(out2.logits[0,-1,:].argmax().item()==t1_clean))
    return float(np.mean(kls)),float(np.mean(t1s))

def mahalanobis_attack(H_q,H_bank,M,query_idx):
    d=H_q.shape[1]
    evals,evecs=torch.linalg.eigh((M+M.T)/2+1e-8*torch.eye(d))
    evals=evals.clamp(min=0)
    L=evecs@torch.diag(torch.sqrt(evals))
    H_q_w=H_q@L
    H_b_w=H_bank@L
    D=torch.cdist(H_q_w,H_b_w)
    n_q=H_q.shape[0]
    ranks=[]
    for i in range(n_q):
        qi=query_idx[i]
        row=D[i]
        rank=(row<row[qi]).sum().item()+1
        ranks.append(rank)
    return {"top1":sum(1 for r in ranks if r==1)/n_q,
            "mrr":sum(1.0/r for r in ranks)/n_q,
            "med_rank":float(np.median(ranks))}

def learn_mahalanobis(H_val,H_bank,val_idx,candidate_Ms):
    best_M=None;best_mrr=0
    for label,M in candidate_Ms.items():
        r=mahalanobis_attack(H_val,H_bank,M,val_idx)
        if r["mrr"]>best_mrr:
            best_mrr=r["mrr"]
            best_M=(label,M)
    return best_M

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    mname=args.model
    dev=args.device
    t0=time.time()

    print("="*60+f"\nOptimal Defense: {mname}\n"+"="*60)

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

    ds=make_ds(tok,n=6000,sl=ctx+1)
    calib=ds[:1000]
    bank=ds[1000:5000]
    val=ds[5000:5500]
    test=ds[5500:6000]

    n_fisher=300 if "mistral" in mname.lower() or "qwen" in mname.lower() else 1000
    print(f"\n--- Computing Fisher matrix F (n={n_fisher}) ---")
    F_mat=compute_fisher_avg(model,calib[:n_fisher],layer,dev,n_cal=n_fisher,ctx=ctx)
    print(f"Fisher trace: {F_mat.trace():.3e}")

    print("\n--- Embedding bank ---")
    H_bank=embed_bank(model,bank,layer,dev,ctx=ctx)
    H_val=embed_bank(model,val,layer,dev,ctx=ctx)
    H_test=embed_bank(model,test,layer,dev,ctx=ctx)

    print("\n--- Computing identity covariance S ---")
    S=compute_id_cov(H_bank,n_pairs=3000)
    print(f"S trace: {S.trace():.3e}")

    print("\n--- Generalized eigendecomposition S u = λ F u ---")
    lambdas,V_gen,F_evals,F_evecs=gen_eigendecomp(S,F_mat,reg_frac=0.01)
    print(f"Top 5 eigenvalues: {lambdas[:5].tolist()}")
    print(f"Bottom 5 eigenvalues: {lambdas[-5:].tolist()}")
    V_gen_norm=V_gen/(V_gen.norm(dim=0)+1e-8)

    print("\n--- Computing grad covariance behavior subspace (baseline) ---")
    sp=Path(f"artifacts/subspace/{mname.replace('/','_')}/layer_{layer}")
    loaded=False
    if sp.exists():
        try:
            evecs_B=torch.load(sp/"grad_evecs.pt",weights_only=True)
            loaded=True
        except: pass
    if not loaded:
        blk_,_,_=get_layer_block(model,layer)
        G_B=torch.zeros(d,d,dtype=torch.float64)
        cnt=0
        for seq in tqdm(calib[:min(1000,len(calib))],desc="grad cov B"):
            x=seq[:ctx].unsqueeze(0).to(dev)
            y=seq[ctx].to(dev) if len(seq)>ctx else seq[-1].to(dev)
            cap=[None]
            def hkc(m,i,o,c=cap):
                c[0]=(o[0] if isinstance(o,tuple) else o).detach()
            h1=blk_.register_forward_hook(hkc)
            with torch.no_grad(): model(x)
            h1.remove()
            h_orig=cap[0][0,-1,:].float()
            h_var=h_orig.clone().requires_grad_(True)
            def inject(m,i,o,hv=h_var):
                oo=o[0] if isinstance(o,tuple) else o
                oo=oo.clone()
                oo[0,-1,:]=hv.to(oo.dtype)
                if isinstance(o,tuple): return (oo,)+o[1:]
                return oo
            h2=blk_.register_forward_hook(inject)
            out=model(x)
            h2.remove()
            logits=out.logits[0,-1,:].float()
            loss=F.cross_entropy(logits.unsqueeze(0),y.unsqueeze(0))
            g=torch.autograd.grad(loss,h_var)[0].detach().float().cpu()
            G_B+=torch.outer(g.to(torch.float64),g.to(torch.float64))
            cnt+=1
        G_B/=cnt
        evals_B,evecs_B=torch.linalg.eigh(G_B.float())
        idx_B=evals_B.argsort(descending=True)
        evecs_B=evecs_B[:,idx_B]
    U_B=evecs_B[:,:k].to(torch.float32)

    H_all=torch.cat([H_bank,H_val],dim=0)
    val_idx=list(range(len(H_bank),len(H_all)))

    print("\n--- Learning Mahalanobis attacker (val) ---")
    U_cpu=U_B.cpu()
    V_id=torch.eye(d)-U_cpu@U_cpu.T
    candidate_Ms={
        "identity":torch.eye(d),
        "id_complement":V_id,
        "gen_eigen_top_k":V_gen[:,:k]@V_gen[:,:k].T,
        "gen_eigen_top_2k":V_gen[:,:2*k]@V_gen[:,:2*k].T,
    }
    best=learn_mahalanobis(H_val,H_all,val_idx,candidate_Ms)
    print(f"Best attacker M: {best[0]}")
    M_attack=best[1]

    def noise_optimal(h,sigma,V=V_gen_norm,k_noise=k):
        z=torch.zeros(d)
        z[:k_noise]=torch.randn(k_noise)*sigma
        noise=V@z
        return h+noise

    def noise_complement(h,sigma,U=U_cpu):
        z=torch.randn_like(h)*sigma
        return h+(z-z@U@U.T)

    def noise_isotropic(h,sigma):
        return h+torch.randn_like(h)*sigma

    evals_F_cached,evecs_F_cached=torch.linalg.eigh(F_mat)
    idx_F=evals_F_cached.argsort(descending=True)
    V_F_I=evecs_F_cached[:,idx_F][:,k:].contiguous()
    def noise_fisher_complement(h,sigma,V_I=V_F_I):
        z=torch.randn(V_I.shape[1])*sigma
        return h+V_I@z

    mechanisms={
        "isotropic":noise_isotropic,
        "complement":noise_complement,
        "fisher_complement":noise_fisher_complement,
        "gen_eigen":noise_optimal,
    }

    sigmas=[0.1,0.3,0.5,1.0,2.0,5.0,10.0]
    results=[]
    H_all_test=torch.cat([H_bank,H_test],dim=0)
    test_idx=list(range(len(H_bank),len(H_all_test)))

    print("\n--- DEFENSE PARETO (Mahalanobis attack) ---")
    for mech_name,mech_fn in mechanisms.items():
        print(f"\n  Mechanism: {mech_name}")
        for sigma in sigmas:
            torch.manual_seed(42)
            H_noisy=torch.stack([mech_fn(H_test[i],sigma) for i in range(len(H_test))])
            r_M=mahalanobis_attack(H_noisy,H_all_test,M_attack,test_idx)
            r_L2=mahalanobis_attack(H_noisy,H_all_test,torch.eye(d),test_idx)
            r_Id=mahalanobis_attack(H_noisy,H_all_test,V_id,test_idx)
            if sigma>0:
                mean_kl,t1=measure_kl(model,test,layer,dev,lambda h,s=sigma,fn=mech_fn:fn(h,s),n_eval=150,ctx=ctx)
            else:
                mean_kl=0;t1=1.0
            entry={
                "mechanism":mech_name,"sigma":sigma,"mean_kl":mean_kl,"t1_agree":t1,
                "attack_M_top1":r_M["top1"],"attack_M_mrr":r_M["mrr"],
                "attack_L2_top1":r_L2["top1"],
                "attack_Id_top1":r_Id["top1"],
                "best_attack_top1":max(r_M["top1"],r_L2["top1"],r_Id["top1"]),
            }
            results.append(entry)
            print(f"    σ={sigma:.1f}: kl={mean_kl:.3f} t1={t1:.3f} M_top1={r_M['top1']:.3f} L2={r_L2['top1']:.3f} Id={r_Id['top1']:.3f} best={entry['best_attack_top1']:.3f}")

    output={
        "model":mname,"layer":layer,"k":k,"d":d,"ctx":ctx,
        "best_attacker":best[0],
        "fisher_trace":float(F_mat.trace()),
        "id_cov_trace":float(S.trace()),
        "top_gen_eigenvalues":lambdas[:10].tolist(),
        "defense":results,
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s":time.time()-t0,
    }
    slug=mname.replace("/","_")
    with open(OUT/f"optimal_defense_{slug}.json","w") as f:
        json.dump(output,f,indent=2)
    print(f"\nResults saved to {OUT}")
    print(f"Total time: {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()

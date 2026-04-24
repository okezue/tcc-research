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
OUT=Path("artifacts/transplant")
OUT.mkdir(parents=True,exist_ok=True)

def make_ds(tok,n=2000,sl=33,seed=42):
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

def compute_grad_cov(model,ds,layer,dev,n_cal=2000,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    d=model.config.hidden_size
    G=torch.zeros(d,d,dtype=torch.float64)
    cnt=0
    for seq in tqdm(ds[:n_cal],desc="grad cov"):
        x=seq[:ctx].unsqueeze(0).to(dev)
        y=seq[ctx].to(dev) if len(seq)>ctx else seq[-1].to(dev)
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
        loss=F.cross_entropy(logits.unsqueeze(0),y.unsqueeze(0))
        g=torch.autograd.grad(loss,h_var)[0].detach().float().cpu()
        G+=torch.outer(g.to(torch.float64),g.to(torch.float64))
        cnt+=1
    G/=cnt
    evals,evecs=torch.linalg.eigh(G.float())
    idx=evals.argsort(descending=True)
    return evecs[:,idx],evals[idx]

def get_hidden(model,seq,layer,dev,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    cap=[None]
    def hk(m,i,o,c=cap):
        c[0]=(o[0] if isinstance(o,tuple) else o).detach()
    h=blk.register_forward_hook(hk)
    with torch.no_grad():
        out=model(seq[:ctx].unsqueeze(0).to(dev))
    h.remove()
    return cap[0][0,-1,:].float(),out.logits[0,-1,:].float()

def inject_and_run(model,seq,layer,h_new,dev,ctx=32):
    blk,_,_=get_layer_block(model,layer)
    def inject(m,i,o):
        oo=o[0] if isinstance(o,tuple) else o
        oo=oo.clone()
        oo[0,-1,:]=h_new.to(oo.dtype).to(oo.device)
        if isinstance(o,tuple): return (oo,)+o[1:]
        return oo
    hh=blk.register_forward_hook(inject)
    with torch.no_grad():
        out=model(seq[:ctx].unsqueeze(0).to(dev))
    hh.remove()
    return out.logits[0,-1,:].float()

def project(h,U):
    return h@U@U.T

def kl_softmax(log_p,q):
    return F.kl_div(log_p,q,reduction='sum',log_target=False).item()

def find_pairs(model,ds,layer,dev,U_B,n_pairs=500,ctx=32):
    hidden_all=[];logits_all=[]
    for seq in tqdm(ds[:2*n_pairs+500],desc="collect"):
        h,lg=get_hidden(model,seq,layer,dev,ctx)
        hidden_all.append(h.cpu())
        logits_all.append(F.log_softmax(lg,dim=-1).cpu())
    H=torch.stack(hidden_all)
    L=torch.stack(logits_all)
    D_kl=torch.zeros(len(H),len(H))
    U=U_B.cpu()
    H_I=H-H@U@U.T
    D_id=torch.cdist(H_I,H_I)
    for i in range(len(H)):
        for j in range(len(H)):
            if i==j: D_kl[i,j]=float('inf')
            else:
                D_kl[i,j]=F.kl_div(L[j],L[i].exp(),reduction='sum',log_target=False).item()
    matched=[]
    unmatched=[]
    for i in range(min(len(H),2*n_pairs)):
        order=D_kl[i].argsort()
        for j in order[:20]:
            if j.item()==i: continue
            if D_id[i,j]>D_id.median()*0.5:
                matched.append((i,j.item()))
                break
        rand_j=torch.randint(0,len(H),(1,)).item()
        while rand_j==i: rand_j=torch.randint(0,len(H),(1,)).item()
        unmatched.append((i,rand_j))
    return matched[:n_pairs//2],unmatched[:n_pairs//2],H,L

def transplant_experiment_noised(model,ds,layer,dev,U_B,k,ctx=32,n_pairs=200,sigma=2.0):
    print(f"\n=== NOISED TRANSPLANT (σ={sigma}) ===")
    print(f"Collecting hidden states...")
    matched,unmatched,H_all,L_all=find_pairs(model,ds,layer,dev,U_B,n_pairs=n_pairs,ctx=ctx)
    print(f"  {len(matched)} matched, {len(unmatched)} unmatched")
    U=U_B.to(dev)
    U_cpu=U.cpu()
    H_all_B=H_all@U_cpu@U_cpu.T
    H_all_I=H_all-H_all_B
    lambdas=[0.0,0.5,1.0]
    results={"noised_identity_swap":[],"noised_behavior_swap":[]}
    torch.manual_seed(42)
    print(f"\n=== Noised identity swap: swap P_I A→B, add N(0,σ²P_I) noise to full h ===")
    for pair_type,pairs in [("matched",matched[:100]),("unmatched",unmatched[:100])]:
        for lam in lambdas:
            hybrids_full=[]
            hybrids_I=[]
            kl_to_A=[];kl_to_B=[]
            rank_A_full=[];rank_B_full=[]
            rank_A_id=[];rank_B_id=[]
            for idx,(ia,ib) in enumerate(pairs):
                seq_a=ds[ia]
                h_A=H_all[ia].to(dev)
                h_B=H_all[ib].to(dev)
                h_A_B=project(h_A.unsqueeze(0),U).squeeze(0)
                h_A_I=h_A-h_A_B
                h_B_B=project(h_B.unsqueeze(0),U).squeeze(0)
                h_B_I=h_B-h_B_B
                h_new=h_A_B+(1-lam)*h_A_I+lam*h_B_I
                noise=torch.randn_like(h_new)*sigma
                noise_I=noise-project(noise.unsqueeze(0),U).squeeze(0)
                h_noisy=h_new+noise_I
                logits_new=inject_and_run(model,seq_a,layer,h_noisy,dev,ctx)
                lp_new=F.log_softmax(logits_new,dim=-1).cpu()
                kl_to_A.append(F.kl_div(lp_new,L_all[ia].exp(),reduction='sum',log_target=False).item())
                kl_to_B.append(F.kl_div(lp_new,L_all[ib].exp(),reduction='sum',log_target=False).item())
                hybrids_full.append(h_noisy.cpu())
                hybrids_I.append((h_noisy-project(h_noisy.unsqueeze(0),U).squeeze(0)).cpu())
            H_q_full=torch.stack(hybrids_full)
            H_q_I=torch.stack(hybrids_I)
            D_full=torch.cdist(H_q_full,H_all)
            D_I=torch.cdist(H_q_I,H_all_I)
            for idx,(ia,ib) in enumerate(pairs):
                row_f=D_full[idx];row_i=D_I[idx]
                rank_A_full.append((row_f<row_f[ia]).sum().item()+1)
                rank_B_full.append((row_f<row_f[ib]).sum().item()+1)
                rank_A_id.append((row_i<row_i[ia]).sum().item()+1)
                rank_B_id.append((row_i<row_i[ib]).sum().item()+1)
            entry={
                "lambda":lam,"pair_type":pair_type,"sigma":sigma,
                "kl_to_A_median":float(np.median(kl_to_A)),
                "kl_to_B_median":float(np.median(kl_to_B)),
                "full_top1_A":float(np.mean([1 if r==1 else 0 for r in rank_A_full])),
                "full_top1_B":float(np.mean([1 if r==1 else 0 for r in rank_B_full])),
                "id_top1_A":float(np.mean([1 if r==1 else 0 for r in rank_A_id])),
                "id_top1_B":float(np.mean([1 if r==1 else 0 for r in rank_B_id])),
                "id_rank_A_median":float(np.median(rank_A_id)),
                "id_rank_B_median":float(np.median(rank_B_id)),
            }
            results["noised_identity_swap"].append(entry)
            print(f"  λ={lam} ({pair_type:>9}): KL→A={entry['kl_to_A_median']:.3f} KL→B={entry['kl_to_B_median']:.3f} "
                  f"full_t1(A)={entry['full_top1_A']:.3f} full_t1(B)={entry['full_top1_B']:.3f} "
                  f"id_t1(A)={entry['id_top1_A']:.3f} id_t1(B)={entry['id_top1_B']:.3f}")
    print(f"\n=== Noised behavior swap: swap P_B A→B, add N(0,σ²I) noise to full h ===")
    for pair_type,pairs in [("matched",matched[:100]),("unmatched",unmatched[:100])]:
        for lam in lambdas:
            hybrids_full=[]
            hybrids_I=[]
            kl_to_A=[];kl_to_B=[]
            rank_A_full=[];rank_B_full=[]
            rank_A_id=[];rank_B_id=[]
            for idx,(ia,ib) in enumerate(pairs):
                seq_a=ds[ia]
                h_A=H_all[ia].to(dev)
                h_B=H_all[ib].to(dev)
                h_A_B=project(h_A.unsqueeze(0),U).squeeze(0)
                h_A_I=h_A-h_A_B
                h_B_B=project(h_B.unsqueeze(0),U).squeeze(0)
                h_new=(1-lam)*h_A_B+lam*h_B_B+h_A_I
                noise=torch.randn_like(h_new)*sigma
                h_noisy=h_new+noise
                logits_new=inject_and_run(model,seq_a,layer,h_noisy,dev,ctx)
                lp_new=F.log_softmax(logits_new,dim=-1).cpu()
                kl_to_A.append(F.kl_div(lp_new,L_all[ia].exp(),reduction='sum',log_target=False).item())
                kl_to_B.append(F.kl_div(lp_new,L_all[ib].exp(),reduction='sum',log_target=False).item())
                hybrids_full.append(h_noisy.cpu())
                hybrids_I.append((h_noisy-project(h_noisy.unsqueeze(0),U).squeeze(0)).cpu())
            H_q_full=torch.stack(hybrids_full)
            H_q_I=torch.stack(hybrids_I)
            D_full=torch.cdist(H_q_full,H_all)
            D_I=torch.cdist(H_q_I,H_all_I)
            for idx,(ia,ib) in enumerate(pairs):
                row_f=D_full[idx];row_i=D_I[idx]
                rank_A_full.append((row_f<row_f[ia]).sum().item()+1)
                rank_B_full.append((row_f<row_f[ib]).sum().item()+1)
                rank_A_id.append((row_i<row_i[ia]).sum().item()+1)
                rank_B_id.append((row_i<row_i[ib]).sum().item()+1)
            entry={
                "lambda":lam,"pair_type":pair_type,"sigma":sigma,
                "kl_to_A_median":float(np.median(kl_to_A)),
                "kl_to_B_median":float(np.median(kl_to_B)),
                "full_top1_A":float(np.mean([1 if r==1 else 0 for r in rank_A_full])),
                "full_top1_B":float(np.mean([1 if r==1 else 0 for r in rank_B_full])),
                "id_top1_A":float(np.mean([1 if r==1 else 0 for r in rank_A_id])),
                "id_top1_B":float(np.mean([1 if r==1 else 0 for r in rank_B_id])),
                "id_rank_A_median":float(np.median(rank_A_id)),
                "id_rank_B_median":float(np.median(rank_B_id)),
            }
            results["noised_behavior_swap"].append(entry)
            print(f"  λ={lam} ({pair_type:>9}): KL→A={entry['kl_to_A_median']:.3f} KL→B={entry['kl_to_B_median']:.3f} "
                  f"full_t1(A)={entry['full_top1_A']:.3f} full_t1(B)={entry['full_top1_B']:.3f} "
                  f"id_t1(A)={entry['id_top1_A']:.3f} id_t1(B)={entry['id_top1_B']:.3f}")
    return results

def transplant_experiment(model,ds,layer,dev,U_B,k,ctx=32,n_pairs=200):
    print(f"\n=== Collecting {2*n_pairs+500} hidden states ===")
    matched,unmatched,H_all,L_all=find_pairs(model,ds,layer,dev,U_B,n_pairs=n_pairs,ctx=ctx)
    print(f"  Matched pairs (low-KL, high-id): {len(matched)}")
    print(f"  Unmatched pairs (random): {len(unmatched)}")

    U=U_B.to(dev)
    lambdas=[0.0,0.25,0.5,0.75,1.0]

    results={"identity_swap":[],"behavior_swap":[]}

    print(f"\n=== Identity swap: h_λ = P_B h_A + (1-λ) P_I h_A + λ P_I h_B ===")
    for pair_type,pairs in [("matched",matched[:100]),("unmatched",unmatched[:100])]:
        for lam in lambdas:
            kl_to_A=[];kl_to_B=[];t1_A=[];t1_B=[]
            for idx,(ia,ib) in enumerate(pairs):
                seq_a=ds[ia]
                h_A=H_all[ia].to(dev)
                h_B=H_all[ib].to(dev)
                h_A_B=project(h_A.unsqueeze(0),U).squeeze(0)
                h_A_I=h_A-h_A_B
                h_B_B=project(h_B.unsqueeze(0),U).squeeze(0)
                h_B_I=h_B-h_B_B
                h_new=h_A_B+(1-lam)*h_A_I+lam*h_B_I
                logits_new=inject_and_run(model,seq_a,layer,h_new,dev,ctx)
                lp_new=F.log_softmax(logits_new,dim=-1).cpu()
                lp_A=L_all[ia]
                lp_B=L_all[ib]
                kl_a=F.kl_div(lp_new,lp_A.exp(),reduction='sum',log_target=False).item()
                kl_b=F.kl_div(lp_new,lp_B.exp(),reduction='sum',log_target=False).item()
                t1_new=logits_new.argmax().item()
                t1_Aval=lp_A.argmax().item()
                t1_Bval=lp_B.argmax().item()
                kl_to_A.append(kl_a);kl_to_B.append(kl_b)
                t1_A.append(int(t1_new==t1_Aval));t1_B.append(int(t1_new==t1_Bval))
            results["identity_swap"].append({
                "lambda":lam,"pair_type":pair_type,
                "kl_to_A_median":float(np.median(kl_to_A)),
                "kl_to_B_median":float(np.median(kl_to_B)),
                "t1_agree_A":float(np.mean(t1_A)),
                "t1_agree_B":float(np.mean(t1_B)),
                "n":len(kl_to_A),
            })
            print(f"  λ={lam} ({pair_type:>9}): KL→A={np.median(kl_to_A):.3f} KL→B={np.median(kl_to_B):.3f} t1(A)={np.mean(t1_A):.3f} t1(B)={np.mean(t1_B):.3f}")

    print(f"\n=== Behavior swap: h_λ = (1-λ) P_B h_A + λ P_B h_B + P_I h_A ===")
    for pair_type,pairs in [("matched",matched[:100]),("unmatched",unmatched[:100])]:
        for lam in lambdas:
            kl_to_A=[];kl_to_B=[];t1_A=[];t1_B=[]
            H_test_full=[]
            H_test_id=[]
            for idx,(ia,ib) in enumerate(pairs):
                seq_a=ds[ia]
                h_A=H_all[ia].to(dev)
                h_B=H_all[ib].to(dev)
                h_A_B=project(h_A.unsqueeze(0),U).squeeze(0)
                h_A_I=h_A-h_A_B
                h_B_B=project(h_B.unsqueeze(0),U).squeeze(0)
                h_new=(1-lam)*h_A_B+lam*h_B_B+h_A_I
                H_test_full.append(h_new.cpu())
                H_test_id.append((h_new-project(h_new.unsqueeze(0),U).squeeze(0)).cpu())
                logits_new=inject_and_run(model,seq_a,layer,h_new,dev,ctx)
                lp_new=F.log_softmax(logits_new,dim=-1).cpu()
                lp_A=L_all[ia]
                lp_B=L_all[ib]
                kl_a=F.kl_div(lp_new,lp_A.exp(),reduction='sum',log_target=False).item()
                kl_b=F.kl_div(lp_new,lp_B.exp(),reduction='sum',log_target=False).item()
                t1_new=logits_new.argmax().item()
                kl_to_A.append(kl_a);kl_to_B.append(kl_b)
                t1_A.append(int(t1_new==lp_A.argmax().item()))
                t1_B.append(int(t1_new==lp_B.argmax().item()))

            H_test_full=torch.stack(H_test_full)
            H_test_id=torch.stack(H_test_id)
            U_cpu=U.cpu()
            H_all_I=H_all-H_all@U_cpu@U_cpu.T
            D_id_attack=torch.cdist(H_test_id,H_all_I)
            rank_A_id=[];rank_B_id=[]
            for idx,(ia,ib) in enumerate(pairs):
                row=D_id_attack[idx]
                rank_A_id.append((row<row[ia]).sum().item()+1)
                rank_B_id.append((row<row[ib]).sum().item()+1)
            results["behavior_swap"].append({
                "lambda":lam,"pair_type":pair_type,
                "kl_to_A_median":float(np.median(kl_to_A)),
                "kl_to_B_median":float(np.median(kl_to_B)),
                "t1_agree_A":float(np.mean(t1_A)),
                "t1_agree_B":float(np.mean(t1_B)),
                "id_rank_A_median":float(np.median(rank_A_id)),
                "id_rank_B_median":float(np.median(rank_B_id)),
                "id_top1_A":float(np.mean([1 if r==1 else 0 for r in rank_A_id])),
                "id_top1_B":float(np.mean([1 if r==1 else 0 for r in rank_B_id])),
                "n":len(kl_to_A),
            })
            print(f"  λ={lam} ({pair_type:>9}): KL→A={np.median(kl_to_A):.3f} KL→B={np.median(kl_to_B):.3f} t1(A)={np.mean(t1_A):.3f} id_top1_A={np.mean([1 if r==1 else 0 for r in rank_A_id]):.3f} id_top1_B={np.mean([1 if r==1 else 0 for r in rank_B_id]):.3f}")

    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--k",type=int,default=128)
    p.add_argument("--n-pairs",type=int,default=200)
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    mname=args.model
    dev=args.device
    t0=time.time()

    print("="*60+f"\nChannel Transplantation: {mname}\n"+"="*60)

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

    ds=make_ds(tok,n=2000,sl=ctx+1)

    sp=Path(f"artifacts/subspace/{mname.replace('/','_')}/layer_{layer}")
    if sp.exists():
        evecs=torch.load(sp/"grad_evecs.pt",weights_only=True)
        print(f"Loaded subspace from {sp}")
    else:
        print(f"Computing subspace at layer {layer}")
        evecs,_=compute_grad_cov(model,ds,layer,dev,n_cal=2000,ctx=ctx)
    U_B=evecs[:,:k].to(torch.float32)

    results=transplant_experiment(model,ds,layer,dev,U_B,k,ctx=ctx,n_pairs=args.n_pairs)
    noised={}
    for sigma in [1.0,2.0,5.0]:
        noised[f"sigma_{sigma}"]=transplant_experiment_noised(model,ds,layer,dev,U_B,k,ctx=ctx,n_pairs=args.n_pairs,sigma=sigma)
    output={"model":mname,"layer":layer,"k":k,"d":d,"ctx":ctx,
            "n_pairs":args.n_pairs,**results,
            "noised":noised,
            "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s":time.time()-t0}
    slug=mname.replace("/","_")
    with open(OUT/f"transplant_{slug}.json","w") as f:
        json.dump(output,f,indent=2)
    print(f"\nResults saved to {OUT}")
    print(f"Total time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

if __name__=="__main__":
    main()

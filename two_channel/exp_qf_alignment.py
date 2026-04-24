#!/usr/bin/env python3
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import json,time,gc,argparse
import numpy as np
from pathlib import Path

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/qf_alignment")
OUT.mkdir(parents=True,exist_ok=True)

def load_subspace(model_slug,layer,k=128):
    sp=Path(f"artifacts/subspace/{model_slug}/layer_{layer}")
    if not sp.exists():
        return None
    evecs=torch.load(sp/"grad_evecs.pt",weights_only=True)
    return evecs[:,:k].to(torch.float32)

def analyze_sae(sae_path,layer,d_in,d_lat,U_B,label):
    from two_channel.exp_scaled_clt import JumpReLUSAE
    sae=JumpReLUSAE(d_in,d_lat)
    sae.load_state_dict(torch.load(sae_path,weights_only=True,map_location="cpu"))
    W=sae.W_dec.data
    arates_path=Path(sae_path).parent.parent/"sae_comparison"/"sae_feature_scores.pt"
    scores=None
    if arates_path.exists():
        scores=torch.load(arates_path,weights_only=True,map_location="cpu")
    q_all=[]
    for f in range(d_lat):
        w=W[f]
        if w.norm()<1e-8: continue
        proj=(w@U_B)
        q=(proj.norm()**2)/(w.norm()**2)
        q_all.append(q.item())
    return q_all

def analyze_pretrained_sae(hf_id,U_B,dev):
    from two_channel.exp_scaled_clt import TopKSAE,load_pretrained_sae
    sae,d_sae,layers=load_pretrained_sae(hf_id,dev)
    sae=sae.cpu()
    W=sae.w_dec.data
    q_all=[]
    for f in range(d_sae):
        w=W[f]
        if w.norm()<1e-8: continue
        U=U_B.cpu()
        proj=(w.float()@U)
        q=(proj.norm()**2)/(w.float().norm()**2)
        q_all.append(q.item())
    return q_all

def analyze_sae_with_scores(sae_json_path,U_B):
    r=json.load(open(sae_json_path))
    per_layer=r.get("sae_per_layer",r.get("per_layer",{}))
    d_lat=r.get("d_latent",r.get("config",{}).get("n",0))
    ckpt_dir=Path(sae_json_path).parent/"sae_weights"
    if not ckpt_dir.exists():
        ckpt_dir=Path(sae_json_path).parent/"checkpoints_standalone"
    results={}
    for layer_str,info in per_layer.items():
        layer=int(layer_str)
        sp=ckpt_dir/f"sae_layer_{layer}.pt"
        if not sp.exists():
            alt=ckpt_dir/f"layer_{layer}.pt"
            if alt.exists(): sp=alt
            else: continue
        from two_channel.exp_scaled_clt import JumpReLUSAE
        d_in=U_B.shape[0]
        sae=JumpReLUSAE(d_in,d_lat)
        try:
            sae.load_state_dict(torch.load(sp,weights_only=True,map_location="cpu"))
        except:
            continue
        W=sae.W_dec.data
        bb_qf=[]
        sc_qf=[]
        U=U_B.cpu()
        for f in range(d_lat):
            w=W[f]
            if w.norm()<1e-8: continue
            proj=(w@U)
            q=(proj.norm()**2)/(w.norm()**2)
            bb_qf.append(q.item())
        results[layer_str]={"all_qf":bb_qf,"n_features":len(bb_qf)}
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    dev=args.device
    t0=time.time()

    print("="*60+"\nq_f Alignment Analysis\n"+"="*60)

    print("\n--- GPT-2 Small ---")
    U_gpt2=load_subspace("openai-community_gpt2",11,k=128)
    if U_gpt2 is not None:
        print(f"Loaded behavior subspace: {U_gpt2.shape}")
        sae_dir=Path("artifacts/sae_comparison")
        sae_json=sae_dir/"sae_comparison.json"
        if sae_json.exists():
            r=json.load(open(sae_json))
            per_layer=r["sae_per_layer"]
            sae_w_dir=sae_dir/"sae_weights"
            gpt2_results={}
            for layer_str,info in per_layer.items():
                layer=int(layer_str)
                sp=sae_w_dir/f"sae_layer_{layer}.pt"
                if not sp.exists(): continue
                d_in=768;d_lat=12288
                sd=torch.load(sp,weights_only=True,map_location="cpu")
                W=sd["W_dec"]
                B_avg_path=sae_dir/"sae_feature_scores.pt"
                scores=torch.load(B_avg_path,weights_only=True,map_location="cpu") if B_avg_path.exists() else None
                bb_qf=[]
                sc_qf=[]
                dead=0
                U=U_gpt2.cpu()
                for f in range(d_lat):
                    w=W[f]
                    if w.norm()<1e-8:
                        dead+=1; continue
                    proj=(w@U)
                    q=(proj.norm()**2)/(w.norm()**2)
                    is_bb=False
                    if scores is not None:
                        B=scores.get("B_avg",None)
                        ar=scores.get("activation_rates",None)
                        if B is not None and ar is not None:
                            if ar[layer,f]>0.001 and B[layer,f]>0:
                                is_bb=True
                    if is_bb:
                        bb_qf.append(q.item())
                    else:
                        if scores is not None and ar is not None and ar[layer,f]>0.001:
                            sc_qf.append(q.item())
                if bb_qf and sc_qf:
                    from sklearn.metrics import roc_auc_score
                    labels=[1]*len(bb_qf)+[0]*len(sc_qf)
                    vals=bb_qf+sc_qf
                    auc=roc_auc_score(labels,vals)
                    gpt2_results[layer_str]={
                        "n_backbone":len(bb_qf),"n_scaffold":len(sc_qf),
                        "bb_qf_mean":float(np.mean(bb_qf)),"bb_qf_median":float(np.median(bb_qf)),
                        "sc_qf_mean":float(np.mean(sc_qf)),"sc_qf_median":float(np.median(sc_qf)),
                        "auroc":auc,"dead":dead,
                    }
                    print(f"  L{layer}: bb_qf={np.mean(bb_qf):.4f} sc_qf={np.mean(sc_qf):.4f} AUROC={auc:.3f} (n_bb={len(bb_qf)} n_sc={len(sc_qf)})")
                else:
                    gpt2_results[layer_str]={"n_backbone":len(bb_qf),"n_scaffold":len(sc_qf),"note":"insufficient data"}
                    print(f"  L{layer}: insufficient bb/sc data (bb={len(bb_qf)} sc={len(sc_qf)})")
            with open(OUT/"qf_gpt2.json","w") as f:
                json.dump(gpt2_results,f,indent=2)
        else:
            print("  SAE comparison JSON not found")
    else:
        print("  Subspace not found")

    print("\n--- Mistral-7B (pretrained SAE) ---")
    mistral_sub=Path("artifacts/subspace/mistralai_Mistral-7B-v0.1/layer_16")
    if mistral_sub.exists():
        evecs=torch.load(mistral_sub/"grad_evecs.pt",weights_only=True)
        U_mistral=evecs[:,:128].to(torch.float32)
        print(f"Loaded behavior subspace: {U_mistral.shape}")
        mistral_clt=Path("artifacts/scaled_clt/scaled_clt_results.json")
        if mistral_clt.exists():
            r=json.load(open(mistral_clt))
            print(f"  Using pretrained SAE results")
            q_all=analyze_pretrained_sae("tylercosgrove/mistral-7b-sparse-autoencoder-layer16",U_mistral,dev)
            if q_all:
                print(f"  Total features analyzed: {len(q_all)}")
                print(f"  Mean q_f: {np.mean(q_all):.4f}")
                print(f"  Median q_f: {np.median(q_all):.4f}")
                print(f"  Std q_f: {np.std(q_all):.4f}")
                print(f"  Expected under random (k/d): {128/4096:.4f}")
                with open(OUT/"qf_mistral_all.json","w") as f:
                    json.dump({"q_all":q_all,"mean":float(np.mean(q_all)),"median":float(np.median(q_all)),
                               "std":float(np.std(q_all)),"expected_random":128/4096,
                               "n_features":len(q_all)},f,indent=2)
    else:
        print("  Mistral subspace not found")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()

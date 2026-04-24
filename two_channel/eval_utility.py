import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from .transforms import Transform, QuantStats, compute_quant_stats, HookTransform
from .compute_subspace import load_model, get_layer_block, load_subspace, generate_random_subspace, make_calibration_dataset

def compute_baseline_stats(model, dataset, device, prefix_len=32):
    total_loss=0.0
    total_toks=0
    logits_list=[]
    for toks in tqdm(dataset,desc="Baseline"):
        x=toks[:prefix_len].unsqueeze(0).to(device)
        y=toks[1:prefix_len+1].unsqueeze(0).to(device)
        with torch.no_grad():
            out=model(x)
            logits=out.logits[0]
            loss=F.cross_entropy(logits,y[0],reduction='sum')
            total_loss+=loss.item()
            total_toks+=prefix_len
            logits_list.append(logits.cpu())
    ppl=torch.exp(torch.tensor(total_loss/total_toks)).item()
    return ppl, logits_list

def eval_utility_with_transform(
    model, layer_idx: int, transform: Transform,
    dataset: list, device: str, prefix_len: int=32,
    baseline_logits: list=None
):
    _, abs_idx, _=get_layer_block(model,layer_idx)

    if hasattr(model,'transformer'):
        block=model.transformer.h[abs_idx]
    elif hasattr(model,'model') and hasattr(model.model,'layers'):
        block=model.model.layers[abs_idx]

    hook=HookTransform(transform)
    hook.register(block)

    total_loss=0.0
    total_toks=0
    total_kl=0.0
    top1_match=0
    n_positions=0

    try:
        for i,toks in enumerate(tqdm(dataset,desc=f"Utility {transform.mode} k={transform.U.shape[1]} b={transform.bits}")):
            x=toks[:prefix_len].unsqueeze(0).to(device)
            y=toks[1:prefix_len+1].unsqueeze(0).to(device)
            with torch.no_grad():
                out=model(x)
                logits=out.logits[0]
                loss=F.cross_entropy(logits,y[0],reduction='sum')
                total_loss+=loss.item()
                total_toks+=prefix_len

                if baseline_logits is not None and i<len(baseline_logits):
                    bl=baseline_logits[i].to(logits.device)
                    p_base=F.softmax(bl,dim=-1)
                    lp_base=F.log_softmax(bl,dim=-1)
                    lp_trans=F.log_softmax(logits,dim=-1)
                    kl=(p_base*(lp_base-lp_trans)).sum(dim=-1).mean().item()
                    total_kl+=kl

                    top1_b=bl.argmax(dim=-1)
                    top1_t=logits.argmax(dim=-1)
                    top1_match+=(top1_b==top1_t).sum().item()
                    n_positions+=prefix_len
    finally:
        hook.remove()

    ppl=torch.exp(torch.tensor(total_loss/total_toks)).item()
    avg_kl=total_kl/max(len(dataset),1) if baseline_logits else 0.0
    top1_acc=top1_match/max(n_positions,1) if baseline_logits else 0.0
    return {"ppl":ppl,"kl":avg_kl,"top1_agreement":top1_acc}

def calibrate_quant_stats(model, layer_idx: int, U: torch.Tensor, mode: str,
                           dataset: list, device: str, prefix_len: int=32,
                           max_samples: int=2000):
    _, abs_idx, _=get_layer_block(model,layer_idx)
    hs_idx=abs_idx+1
    all_proj=[]
    for toks in dataset[:max_samples]:
        x=toks[:prefix_len].unsqueeze(0).to(device)
        with torch.no_grad():
            out=model(x,output_hidden_states=True)
            h=out.hidden_states[hs_idx][0]
        if mode=="behavior":
            p=h@U.to(h.device)@U.to(h.device).T
        elif mode=="identity":
            p=h-h@U.to(h.device)@U.to(h.device).T
        elif mode=="random":
            p=h@U.to(h.device)@U.to(h.device).T
        else:
            p=h
        all_proj.append(p.cpu())
    stacked=torch.cat(all_proj,dim=0)
    return compute_quant_stats(stacked)

def run_utility_grid(
    model_id: str, layers: list, k_values: list,
    bits_values: list, sigma_values: list,
    device: str="mps", prefix_len: int=32,
    n_eval: int=2000, n_cal: int=2000,
    subspace_dir: str="artifacts/subspace"
):
    model,tok=load_model(model_id,device)
    eval_ds=make_calibration_dataset(tok,n=n_eval,seq_len=prefix_len+1,seed=123)
    cal_ds=make_calibration_dataset(tok,n=n_cal,seq_len=prefix_len+1,seed=456)

    print("Computing baseline...")
    base_ppl,base_logits=compute_baseline_stats(model,eval_ds,device,prefix_len)
    print(f"Baseline PPL: {base_ppl:.4f}")

    results=[]
    d=model.config.hidden_size

    for li in layers:
        _,abs_idx,_=get_layer_block(model,li)
        sp=Path(subspace_dir)/model_id.replace("/","_")/f"layer_{abs_idx}"

        for k in k_values:
            U_grad,evals=load_subspace(sp,k,mode="grad")
            U_pca,_=load_subspace(sp,k,mode="act")
            U_rand=generate_random_subspace(d,k)

            energy_frac=evals.sum()/torch.load(sp/"grad_evals.pt",weights_only=True).sum()

            for mode,U,label in [
                ("behavior",U_grad,"grad"),
                ("identity",U_grad,"grad"),
                ("random",U_rand,"random"),
                ("full",U_grad,"full"),
            ]:
                for b in bits_values:
                    for sigma in sigma_values:
                        stats=calibrate_quant_stats(model,li,U.to(device),mode,cal_ds,device,prefix_len,max_samples=500) if b<32 else None
                        t=Transform(U.to(device),mode=mode,bits=b,sigma=sigma,stats=stats)
                        r=eval_utility_with_transform(model,li,t,eval_ds,device,prefix_len,base_logits)
                        entry={
                            "layer":abs_idx,"k":k,"mode":mode,"subspace":label,
                            "bits":b,"sigma":sigma,
                            "ppl":r["ppl"],"dppl":r["ppl"]-base_ppl,
                            "kl":r["kl"],"top1":r["top1_agreement"],
                            "energy_frac":energy_frac.item() if mode!="full" else 1.0,
                        }
                        results.append(entry)
                        print(f"  L={abs_idx} k={k} {mode}/{label} b={b} s={sigma:.2f}: PPL={r['ppl']:.2f} dPPL={entry['dppl']:.2f} KL={r['kl']:.4f} top1={entry['top1']:.3f}")

    return results, base_ppl

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model-id",default="openai-community/gpt2")
    p.add_argument("--layers",nargs="+",type=int,default=[6,-1])
    p.add_argument("--k-values",nargs="+",type=int,default=[32,64,128,256])
    p.add_argument("--bits",nargs="+",type=int,default=[32,16,8,6,4])
    p.add_argument("--sigma",nargs="+",type=float,default=[0.0])
    p.add_argument("--device",default="mps")
    p.add_argument("--n-eval",type=int,default=2000)
    p.add_argument("--out",default="artifacts/results/utility.json")
    args=p.parse_args()

    results,base=run_utility_grid(
        args.model_id,args.layers,args.k_values,
        args.bits,args.sigma,args.device,
        n_eval=args.n_eval
    )

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,"w") as f:
        json.dump({"baseline_ppl":base,"results":results},f,indent=2)
    print(f"Saved utility results to {args.out}")

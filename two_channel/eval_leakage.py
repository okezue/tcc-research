import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, DynamicCache
from pathlib import Path
from tqdm import tqdm
import json
import argparse
from .transforms import Transform, QuantStats, compute_quant_stats
from .compute_subspace import load_model, get_layer_block, load_subspace, generate_random_subspace, make_calibration_dataset

def _expand_cache(kv, batch_size):
    expanded=DynamicCache()
    for i,layer in enumerate(kv):
        k,v=layer[0],layer[1]
        expanded.update(k.expand(batch_size,-1,-1,-1),v.expand(batch_size,-1,-1,-1),i)
    return expanded

def vocab_sweep_cached(
    model, tokenizer, layer_idx: int,
    prefix_ids: torch.Tensor, true_next: int,
    transform: Transform, device: str,
    batch_size: int=512
):
    _, abs_idx, _=get_layer_block(model,layer_idx)
    hs_idx=abs_idx+1

    x=prefix_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out=model(x,use_cache=True,output_hidden_states=True)
        kv=out.past_key_values

    true_id=torch.tensor([[true_next]],device=device)
    with torch.no_grad():
        out_true=model(true_id,past_key_values=kv,output_hidden_states=True)
        h_true=out_true.hidden_states[hs_idx][0,-1,:]

    h_true_t=transform(h_true.unsqueeze(0)).squeeze(0)

    V=tokenizer.vocab_size
    all_ids=torch.arange(V,device=device)
    min_dist=float('inf')
    collision_count=0
    dists=[]

    has_codes=transform.bits<32 and transform.stats is not None
    if has_codes:
        true_code=transform.quantize_to_codes(h_true.unsqueeze(0)).squeeze(0)

    for start in range(0,V,batch_size):
        end=min(start+batch_size,V)
        batch_ids=all_ids[start:end].unsqueeze(1)
        b=batch_ids.shape[0]

        kv_expanded=_expand_cache(kv,b)

        with torch.no_grad():
            out_batch=model(batch_ids,past_key_values=kv_expanded,output_hidden_states=True)
            h_batch=out_batch.hidden_states[hs_idx][:,-1,:]

        h_batch_t=transform(h_batch)
        d=torch.norm(h_batch_t-h_true_t.unsqueeze(0),dim=-1)
        dists.append(d.cpu())

        skip_mask=all_ids[start:end]==true_next
        d_other=d.clone()
        d_other[skip_mask]=float('inf')
        batch_min=d_other.min().item()
        if batch_min<min_dist:
            min_dist=batch_min

        if has_codes:
            codes_batch=transform.quantize_to_codes(h_batch)
            match=(codes_batch==true_code.unsqueeze(0)).all(dim=-1)
            collision_count+=match.sum().item()

    all_dists=torch.cat(dists,dim=0)

    if not has_codes:
        collision_count=(all_dists<1e-6).sum().item()

    true_rank=(all_dists<all_dists[true_next]).sum().item()+1

    return {
        "margin":min_dist,
        "collision_count":collision_count,
        "true_rank":true_rank,
    }

def eval_leakage_grid(
    model_id: str, layers: list, k_values: list,
    bits_values: list, device: str="mps",
    prefix_len: int=16, n_prefixes: int=100,
    batch_size: int=512,
    subspace_dir: str="artifacts/subspace"
):
    model,tok=load_model(model_id,device)
    ds=make_calibration_dataset(tok,n=n_prefixes+100,seq_len=prefix_len+1,seed=789)
    ds=ds[:n_prefixes]
    d=model.config.hidden_size

    results=[]

    for li in layers:
        _,abs_idx,_=get_layer_block(model,li)
        sp=Path(subspace_dir)/model_id.replace("/","_")/f"layer_{abs_idx}"
        cal_ds=make_calibration_dataset(tok,n=500,seq_len=prefix_len+1,seed=999)

        for k in k_values:
            U_grad,_=load_subspace(sp,k,mode="grad")
            U_rand=generate_random_subspace(d,k)

            for mode,U,label in [
                ("behavior",U_grad,"grad"),
                ("identity",U_grad,"grad"),
                ("random",U_rand,"random"),
                ("full",U_grad,"full"),
            ]:
                for b in bits_values:
                    stats=None
                    if b<32:
                        from .eval_utility import calibrate_quant_stats
                        stats=calibrate_quant_stats(model,li,U.to(device),mode,cal_ds,device,prefix_len,max_samples=300)

                    t=Transform(U.to(device),mode=mode,bits=b,sigma=0.0,stats=stats)

                    margins=[]
                    collisions=[]
                    ranks=[]
                    for j,toks in enumerate(tqdm(ds,desc=f"Leakage L={abs_idx} {mode} k={k} b={b}",leave=False)):
                        prefix=toks[:prefix_len]
                        true_next=toks[prefix_len].item()
                        r=vocab_sweep_cached(model,tok,li,prefix,true_next,t,device,batch_size)
                        margins.append(r["margin"])
                        collisions.append(r["collision_count"])
                        ranks.append(r["true_rank"])

                    mt=torch.tensor(margins)
                    ct=torch.tensor(collisions,dtype=torch.float)
                    rt=torch.tensor(ranks,dtype=torch.float)

                    entry={
                        "layer":abs_idx,"k":k,"mode":mode,"subspace":label,
                        "bits":b,
                        "margin_median":mt.median().item(),
                        "margin_p10":mt.quantile(0.1).item(),
                        "margin_p90":mt.quantile(0.9).item(),
                        "collision_median":ct.median().item(),
                        "log2_collision_median":torch.log2(ct.clamp(min=1)).median().item(),
                        "unique_frac":(ct==1).float().mean().item(),
                        "rank_median":rt.median().item(),
                    }
                    results.append(entry)
                    print(f"  L={abs_idx} k={k} {mode}/{label} b={b}: margin_med={entry['margin_median']:.4f} coll_med={entry['collision_median']:.0f} unique={entry['unique_frac']:.3f}")

    return results

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model-id",default="openai-community/gpt2")
    p.add_argument("--layers",nargs="+",type=int,default=[6,-1])
    p.add_argument("--k-values",nargs="+",type=int,default=[32,64,128,256])
    p.add_argument("--bits",nargs="+",type=int,default=[16,8,6,4])
    p.add_argument("--n-prefixes",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--device",default="mps")
    p.add_argument("--out",default="artifacts/results/leakage.json")
    args=p.parse_args()

    results=eval_leakage_grid(
        args.model_id,args.layers,args.k_values,
        args.bits,args.device,
        n_prefixes=args.n_prefixes,
        batch_size=args.batch_size
    )

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,"w") as f:
        json.dump({"results":results},f,indent=2)
    print(f"Saved leakage results to {args.out}")

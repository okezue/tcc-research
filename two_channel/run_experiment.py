#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="Two-Channel Experiment Runner")
    p.add_argument("--model-id",default="openai-community/gpt2")
    p.add_argument("--layers",nargs="+",type=int,default=[6,-1])
    p.add_argument("--k-values",nargs="+",type=int,default=[32,64,128,256])
    p.add_argument("--bits",nargs="+",type=int,default=[32,16,8,6,4])
    p.add_argument("--sigma",nargs="+",type=float,default=[0.0])
    p.add_argument("--device",default="mps")
    p.add_argument("--n-cal",type=int,default=5000)
    p.add_argument("--n-eval",type=int,default=2000)
    p.add_argument("--n-prefixes",type=int,default=100)
    p.add_argument("--prefix-len",type=int,default=32)
    p.add_argument("--batch-size",type=int,default=512)
    p.add_argument("--out-dir",default="artifacts")
    p.add_argument("--skip-subspace",action="store_true")
    p.add_argument("--skip-utility",action="store_true")
    p.add_argument("--skip-leakage",action="store_true")
    p.add_argument("--skip-plots",action="store_true")
    args=p.parse_args()

    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)

    t0=time.time()

    if not args.skip_subspace:
        print("="*60)
        print("STEP 1: Computing behavior subspaces")
        print("="*60)
        from two_channel.compute_subspace import load_model, make_calibration_dataset, compute_gradient_covariance, save_subspace
        model,tok=load_model(args.model_id,args.device)
        ds=make_calibration_dataset(tok,n=args.n_cal,seq_len=args.prefix_len+1)
        for li in args.layers:
            res=compute_gradient_covariance(
                model,tok,li,ds,
                prefix_len=args.prefix_len,
                device=args.device,
                max_samples=args.n_cal
            )
            save_subspace(res,out/"subspace",args.model_id,res["layer_idx"])
        del model
        import torch; torch.mps.empty_cache() if args.device=="mps" else None
        print(f"Subspace computation done in {time.time()-t0:.0f}s\n")

    if not args.skip_utility:
        print("="*60)
        print("STEP 2: Utility evaluation")
        print("="*60)
        from two_channel.eval_utility import run_utility_grid
        t1=time.time()
        util_results,base_ppl=run_utility_grid(
            args.model_id,args.layers,args.k_values,
            args.bits,args.sigma,args.device,
            prefix_len=args.prefix_len,
            n_eval=args.n_eval,
            subspace_dir=str(out/"subspace")
        )
        (out/"results").mkdir(parents=True,exist_ok=True)
        with open(out/"results/utility.json","w") as f:
            json.dump({"baseline_ppl":base_ppl,"results":util_results},f,indent=2)
        print(f"Utility eval done in {time.time()-t1:.0f}s\n")

    if not args.skip_leakage:
        print("="*60)
        print("STEP 3: Leakage / margin evaluation")
        print("="*60)
        from two_channel.eval_leakage import eval_leakage_grid
        t2=time.time()
        leak_bits=[b for b in args.bits if b<32]
        if not leak_bits:
            leak_bits=[16,8,6,4]
        leak_results=eval_leakage_grid(
            args.model_id,args.layers,args.k_values,
            leak_bits,args.device,
            prefix_len=min(args.prefix_len,16),
            n_prefixes=args.n_prefixes,
            batch_size=args.batch_size,
            subspace_dir=str(out/"subspace")
        )
        with open(out/"results/leakage.json","w") as f:
            json.dump({"results":leak_results},f,indent=2)
        print(f"Leakage eval done in {time.time()-t2:.0f}s\n")

    if not args.skip_plots:
        print("="*60)
        print("STEP 4: Generating plots")
        print("="*60)
        from two_channel.plot_results import generate_all
        import torch
        from two_channel.compute_subspace import get_layer_block, load_model
        model,_=load_model(args.model_id,"cpu")
        actual_layers=[]
        for li in args.layers:
            _,ai,_=get_layer_block(model,li)
            actual_layers.append(ai)
        del model

        generate_all(
            str(out/"results/utility.json"),
            str(out/"results/leakage.json"),
            str(out/"subspace"),
            args.model_id,
            actual_layers,
            args.k_values,
            str(out/"plots")
        )

    total=time.time()-t0
    print(f"\nTotal experiment time: {total:.0f}s ({total/60:.1f}min)")

if __name__=="__main__":
    main()

#!/usr/bin/env python3
"""
Section 1: Baseline SipIt reproduction (verify exact inversion works)
Section 7: End-to-end SipIt inversion under transforms
Scaled CLT: Retrain CLT at 10M tokens
"""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sipit_root=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..','SIPIT')
if os.path.exists(sipit_root):
    sys.path.insert(0,sipit_root)
clt_root=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..','CLT-Forge','src')
if os.path.exists(clt_root):
    sys.path.insert(0,clt_root)
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
OUT=Path("artifacts/remaining")
OUT.mkdir(parents=True,exist_ok=True)

def section1_baseline_sipit():
    print("="*60+"\nSECTION 1: Baseline SipIt Reproduction\n"+"="*60)
    try:
        from src.utils.model import setup, hidden_states_from_input_ids, hidden_states_from_input_ids_iterative
        from src.algorithm.SIPIT import SIPIT
        from src.algorithm.BruteForce import BruteForce
    except ImportError:
        print("  SipIt not importable, using standalone implementation")
        section1_standalone()
        return

    model,tokenizer,device,layer_idx=setup("openai-community/gpt2",precision=32,layer_idx=-1)

    prompts=[
        "Hello world",
        "The capital of France is",
        "def fibonacci(n):",
        "In 1776, the",
        "import numpy",
    ]

    s1_dir=OUT/"section1"
    s1_dir.mkdir(parents=True,exist_ok=True)
    results=[]

    for method_name,MethodClass in [("SIPIT",SIPIT),("BruteForce",BruteForce)]:
        for prompt in prompts:
            print(f"  {method_name} on: {prompt}")
            algo=MethodClass()
            input_ids=tokenizer(prompt,return_tensors="pt",add_special_tokens=False)["input_ids"][0].to(device)
            t0=time.time()
            try:
                match,inv_time,timesteps,times=algo.inversion_attack(
                    input_ids=input_ids,model=model,tokenizer=tokenizer,
                    layer_idx=layer_idx,step_size=1.0,seed=42
                )
                elapsed=time.time()-t0
                entry={"method":method_name,"prompt":prompt,"match":bool(match),
                       "time":elapsed,"n_tokens":len(input_ids)}
                results.append(entry)
                status="EXACT MATCH" if match else "FAILED"
                print(f"    {status} in {elapsed:.1f}s")
            except Exception as e:
                print(f"    Error: {e}")
                results.append({"method":method_name,"prompt":prompt,"match":False,"error":str(e)})

            if method_name=="BruteForce" and len(input_ids)>3:
                print(f"    (skipping BruteForce for long prompts)")
                break

    with open(s1_dir/"baseline_results.json","w") as f:
        json.dump(results,f,indent=2)

    n_match=sum(1 for r in results if r.get("match"))
    n_total=len(results)
    print(f"\n  Baseline: {n_match}/{n_total} exact inversions")
    print(f"  Results saved to {s1_dir}")

def section1_standalone():
    print("  Running standalone baseline inversion test...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model=AutoModelForCausalLM.from_pretrained("openai-community/gpt2",output_hidden_states=True)
    model.eval().to(DEV)
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained("openai-community/gpt2")

    prompts=["Hello world","The capital of France is","def fibonacci(n):","In 1776, the","import numpy"]
    s1_dir=OUT/"section1"
    s1_dir.mkdir(parents=True,exist_ok=True)
    results=[]

    for prompt in prompts:
        ids=tok(prompt,return_tensors="pt",add_special_tokens=False)["input_ids"][0].to(DEV)
        n_tokens=len(ids)

        with torch.no_grad():
            out=model(ids.unsqueeze(0),output_hidden_states=True)
            target_states=[]
            for t in range(1,n_tokens+1):
                o=model(ids[:t].unsqueeze(0),output_hidden_states=True)
                target_states.append(o.hidden_states[-1][0,-1,:])
            target_states=torch.stack(target_states)

        recovered=[]
        V=model.config.vocab_size
        for t in range(n_tokens):
            target_h=target_states[t]
            prefix=torch.tensor(recovered,dtype=torch.long,device=DEV) if recovered else torch.tensor([],dtype=torch.long,device=DEV)

            best_token=None
            best_dist=float('inf')
            batch_size=2048

            for start in range(0,V,batch_size):
                end=min(start+batch_size,V)
                candidates=torch.arange(start,end,device=DEV)
                if len(prefix)>0:
                    seqs=torch.cat([prefix.unsqueeze(0).expand(end-start,-1),candidates.unsqueeze(1)],dim=1)
                else:
                    seqs=candidates.unsqueeze(1)
                with torch.no_grad():
                    o=model(seqs,output_hidden_states=True)
                    h=o.hidden_states[-1][:,-1,:]
                dists=torch.norm(h-target_h.unsqueeze(0),dim=-1)
                min_idx=dists.argmin().item()
                if dists[min_idx].item()<best_dist:
                    best_dist=dists[min_idx].item()
                    best_token=start+min_idx

            recovered.append(best_token)
            match_so_far=recovered==ids[:t+1].cpu().tolist()

        exact_match=(recovered==ids.cpu().tolist())
        per_token_match=sum(1 for a,b in zip(recovered,ids.cpu().tolist()) if a==b)/n_tokens
        results.append({
            "prompt":prompt,"n_tokens":n_tokens,"exact_match":bool(exact_match),
            "per_token_accuracy":per_token_match,"best_dist_last":best_dist,
            "recovered":tok.decode(recovered)
        })
        status="EXACT" if exact_match else f"partial ({per_token_match:.0%})"
        print(f"    '{prompt}' -> {status} | recovered: '{tok.decode(recovered)}'")

    with open(s1_dir/"baseline_results.json","w") as f:
        json.dump(results,f,indent=2)
    n_exact=sum(1 for r in results if r["exact_match"])
    print(f"\n  Baseline: {n_exact}/{len(results)} exact inversions")

def section7_transformed_inversion():
    print("="*60+"\nSECTION 7: End-to-End SipIt Inversion Under Transforms\n"+"="*60)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from two_channel.transforms import Transform, project, project_complement, quantize, QuantStats, compute_quant_stats
    from two_channel.compute_subspace import load_subspace, generate_random_subspace, get_layer_block

    model=AutoModelForCausalLM.from_pretrained("openai-community/gpt2",output_hidden_states=True)
    model.eval().to(DEV)
    for p in model.parameters(): p.requires_grad_(False)
    tok=AutoTokenizer.from_pretrained("openai-community/gpt2")

    prompts=["Hello world","The capital","def fib","In 1776","import np"]
    layer_idx=-1
    abs_layer=model.config.num_hidden_layers

    s7_dir=OUT/"section7"
    s7_dir.mkdir(parents=True,exist_ok=True)

    sp=Path(f"artifacts/subspace/openai-community_gpt2/layer_{abs_layer-1}")
    if not sp.exists():
        sp=Path(f"artifacts/subspace/openai-community_gpt2/layer_11")

    results=[]

    for k in [64,256]:
        U_grad=None
        if (sp/"grad_evecs.pt").exists():
            evecs=torch.load(sp/"grad_evecs.pt",weights_only=True)
            U_grad=evecs[:,:k].to(torch.float32).to(DEV)
        U_rand=generate_random_subspace(768,k).to(DEV)

        configs=[
            ("full",None),
            ("behavior",U_grad),
            ("identity",U_grad),
            ("random",U_rand),
        ]

        for mode,U in configs:
            if U is None and mode!="full": continue

            for prompt in prompts:
                ids=tok(prompt,return_tensors="pt",add_special_tokens=False)["input_ids"][0].to(DEV)
                n_tokens=len(ids)

                def get_target_h(token_ids):
                    with torch.no_grad():
                        o=model(token_ids.unsqueeze(0),output_hidden_states=True)
                        h=o.hidden_states[-1][0,-1,:]
                    if mode=="full": return h
                    elif mode=="behavior": return project(h.unsqueeze(0),U).squeeze(0)
                    elif mode=="identity": return project_complement(h.unsqueeze(0),U).squeeze(0)
                    elif mode=="random": return project(h.unsqueeze(0),U).squeeze(0)
                    return h

                target_states=[]
                for t in range(1,n_tokens+1):
                    target_states.append(get_target_h(ids[:t]))
                target_states=torch.stack(target_states)

                recovered=[]
                V=model.config.vocab_size
                batch_size=2048

                for t in range(n_tokens):
                    target_h=target_states[t]
                    prefix=torch.tensor(recovered,dtype=torch.long,device=DEV)
                    best_token=None; best_dist=float('inf')

                    for start in range(0,V,batch_size):
                        end=min(start+batch_size,V)
                        candidates=torch.arange(start,end,device=DEV)
                        if len(prefix)>0:
                            seqs=torch.cat([prefix.unsqueeze(0).expand(end-start,-1),candidates.unsqueeze(1)],dim=1)
                        else:
                            seqs=candidates.unsqueeze(1)
                        with torch.no_grad():
                            o=model(seqs,output_hidden_states=True)
                            h=o.hidden_states[-1][:,-1,:]
                        if mode=="behavior": h=project(h,U)
                        elif mode=="identity": h=project_complement(h,U)
                        elif mode=="random": h=project(h,U)
                        dists=torch.norm(h-target_h.unsqueeze(0),dim=-1)
                        mi=dists.argmin().item()
                        if dists[mi].item()<best_dist:
                            best_dist=dists[mi].item(); best_token=start+mi
                    recovered.append(best_token)

                exact=(recovered==ids.cpu().tolist())
                per_tok=sum(1 for a,b in zip(recovered,ids.cpu().tolist()) if a==b)/n_tokens
                entry={"mode":mode,"k":k,"prompt":prompt,"exact_match":bool(exact),
                       "per_token_accuracy":per_tok,"recovered":tok.decode(recovered)}
                results.append(entry)
                sym="Y" if exact else "N"
                print(f"    k={k} {mode:10s} '{prompt}' -> {sym} ({per_tok:.0%}) '{tok.decode(recovered)}'")

    with open(s7_dir/"transformed_inversion.json","w") as f:
        json.dump(results,f,indent=2)

    summary={}
    for r in results:
        key=f"{r['mode']}_k{r['k']}"
        if key not in summary: summary[key]={"exact":0,"total":0,"per_tok_sum":0}
        summary[key]["total"]+=1
        summary[key]["exact"]+=int(r["exact_match"])
        summary[key]["per_tok_sum"]+=r["per_token_accuracy"]
    for key,v in summary.items():
        v["exact_rate"]=v["exact"]/v["total"]
        v["mean_per_tok"]=v["per_tok_sum"]/v["total"]

    with open(s7_dir/"inversion_summary.json","w") as f:
        json.dump(summary,f,indent=2)

    fig,ax=plt.subplots(figsize=(10,5))
    keys=sorted(summary.keys())
    exact_rates=[summary[k]["exact_rate"] for k in keys]
    per_tok_rates=[summary[k]["mean_per_tok"] for k in keys]
    x=np.arange(len(keys))
    ax.bar(x-0.2,exact_rates,0.4,label="Exact match rate",color='#3498db')
    ax.bar(x+0.2,per_tok_rates,0.4,label="Per-token accuracy",color='#e74c3c')
    ax.set_xticks(x); ax.set_xticklabels(keys,rotation=45,ha='right')
    ax.set_ylabel("Rate"); ax.set_title("Section 7: Inversion Success Under Transforms")
    ax.legend(); ax.set_ylim(0,1.1)
    fig.tight_layout()
    fig.savefig(s7_dir/"inversion_under_transforms.png",dpi=150)
    plt.close(fig)

    print(f"\n  Summary:")
    for k,v in sorted(summary.items()):
        print(f"    {k}: exact={v['exact_rate']:.0%} per_tok={v['mean_per_tok']:.0%}")

def scaled_clt_training():
    print("="*60+"\nSCALED CLT: 10M token training\n"+"="*60)
    try:
        from clt_forge.config.clt_training_runner_config import CLTTrainingRunnerConfig
        from clt_forge.clt_training_runner import CLTTrainingRunner
    except ImportError:
        print("  CLT-Forge not available, skipping")
        return

    ckpt_path=str(OUT/"clt_scaled/checkpoints")
    Path(ckpt_path).mkdir(parents=True,exist_ok=True)

    total_tok=10_000_000
    batch=256
    steps=total_tok//batch

    cfg=CLTTrainingRunnerConfig(
        device=DEV,
        dtype="bfloat16" if DEV=="cuda" else "float32",
        model_name="gpt2",
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        context_size=32,
        d_in=768,
        d_latent=12288,
        checkpoint_path=ckpt_path,
        train_batch_size_tokens=batch,
        total_training_tokens=total_tok,
        lr=4e-4,
        lr_warm_up_steps=1000,
        lr_decay_steps=max(1,steps-1000),
        l0_coefficient=2.0,
        dead_penalty_coef=1e-5,
        l0_warm_up_steps=int(0.7*steps),
        distributed_setup="None",
        log_to_wandb=False,
        n_batches_in_buffer=16,
        store_batch_size_prompts=32,
        gradient_accumulation_steps=4,
        n_checkpoints=8,
        model_from_pretrained_kwargs={},
    )

    runner=CLTTrainingRunner(cfg,rank=0,world_size=1)
    runner.run()
    print(f"  Scaled CLT training complete. Checkpoints at {ckpt_path}")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--skip",nargs="*",default=[])
    p.add_argument("--only",type=str,default=None)
    args=p.parse_args()

    t0=time.time()
    stages=[
        ("1",section1_baseline_sipit),
        ("7",section7_transformed_inversion),
        ("clt",scaled_clt_training),
    ]

    for sid,fn in stages:
        if args.only and sid!=args.only: continue
        if sid in args.skip: continue
        try:
            fn()
        except Exception as e:
            print(f"Section {sid} failed: {e}")
            import traceback; traceback.print_exc()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\nTotal time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)")

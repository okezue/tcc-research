#!/usr/bin/env python3
"""DP-SGD fine-tune GPT-2 at target eps, then evaluate inversion of held-out inference prompts.

Uses opacus for DP-SGD. Runs at eps in {2,4,8} with delta=1e-6.
Evaluates whether the fine-tuned model's layer-6 hidden states still leak
identifiable prefix information via retrieval (it should — DP-SGD protects
training data, not inference activations).
"""
import os,sys,json,argparse,time,math
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from pathlib import Path
from tqdm import tqdm

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/dp_sgd")
OUT.mkdir(parents=True,exist_ok=True)

def load_data(tok,n,sl,seed=42):
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

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--eps",type=float,default=8.0)
    p.add_argument("--delta",type=float,default=1e-6)
    p.add_argument("--n_train",type=int,default=10000)
    p.add_argument("--n_eval",type=int,default=2000)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch_size",type=int,default=64)
    p.add_argument("--max_grad_norm",type=float,default=1.0)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--seq_len",type=int,default=64)
    p.add_argument("--layer",type=int,default=6)
    args=p.parse_args()
    t0=time.time()

    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(args.model).to(DEV)
    model.train()
    d=model.config.hidden_size

    print(f"[dp_sgd] target eps={args.eps} delta={args.delta} n_train={args.n_train}")
    ds=load_data(tok,n=args.n_train+args.n_eval,sl=args.seq_len)
    X=torch.stack(ds[:args.n_train]).to(DEV)

    try:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
    except ImportError:
        os.system("pip install --break-system-packages --quiet opacus")
        from opacus import PrivacyEngine

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    class DS(torch.utils.data.Dataset):
        def __init__(s,x): s.x=x
        def __len__(s): return len(s.x)
        def __getitem__(s,i): return s.x[i]
    loader=torch.utils.data.DataLoader(DS(X),batch_size=args.batch_size,shuffle=True)

    pe=PrivacyEngine()
    model,opt,loader=pe.make_private_with_epsilon(
        module=model,optimizer=opt,data_loader=loader,
        target_epsilon=args.eps,target_delta=args.delta,
        epochs=args.epochs,max_grad_norm=args.max_grad_norm)

    step=0
    for epoch in range(args.epochs):
        for batch in tqdm(loader,desc=f"ep{epoch}"):
            b=batch.to(DEV)
            out=model(b,labels=b)
            loss=out.loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            step+=1
            if step%50==0:
                print(f"  step {step} loss={loss.item():.3f}")
    eps_spent=pe.get_epsilon(args.delta)
    print(f"final eps={eps_spent:.2f} (target {args.eps})")

    # Evaluate: does the DP-SGD model still leak inference prefixes via retrieval at layer L?
    model.eval()
    from two_channel.exp_optimal_defense import embed_bank
    ds_eval=ds[args.n_train:args.n_train+args.n_eval]
    print(f"evaluating retrieval on {len(ds_eval)} held-out prefixes...")
    H=embed_bank(model,ds_eval,args.layer,DEV,ctx=args.seq_len-1)
    from two_channel.mahalanobis_attacker import l2_retrieval
    n_q=min(500,len(H)//2)
    bank=list(range(len(H)))
    H_q=H[:n_q]
    idx=list(range(n_q))
    H_all=H
    r=l2_retrieval(H_q,H_all,idx)
    print(f"retrieval top-1 on DP-SGD GPT-2 held-out prefixes: {r['top1']:.3f}")

    out={"model":args.model,"target_eps":args.eps,"delta":args.delta,
         "actual_eps":eps_spent,"n_train":args.n_train,"epochs":args.epochs,
         "held_out_retrieval_top1":r["top1"],"elapsed_s":time.time()-t0}
    with open(OUT/f"dp_sgd_eps{args.eps:.0f}.json","w") as f: json.dump(out,f,indent=2)
    print(f"saved -> {OUT}/dp_sgd_eps{args.eps:.0f}.json")

if __name__=="__main__": main()

#!/usr/bin/env python3
import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import json,time,gc,argparse,math
import numpy as np
from pathlib import Path
from tqdm import tqdm

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/scale_results")
OUT.mkdir(parents=True,exist_ok=True)

MODELS=[
    ("openai-community/gpt2-large","GPT-2 Large","Dense","774M",1280,36,18),
    ("openai-community/gpt2-xl","GPT-2 XL","Dense","1.5B",1600,48,24),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0","TinyLlama-1.1B","Dense","1.1B",2048,22,11),
    ("microsoft/phi-2","Phi-2","Dense","2.7B",2560,32,16),
    ("Qwen/Qwen2.5-3B","Qwen2.5-3B","Dense","3.1B",2048,36,18),
]

def make_ds(tok,n=2500,sl=33,seed=42):
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
    if hasattr(model,'transformer'):
        if hasattr(model.transformer,'h'):
            return model.transformer.h[layer],layer,len(model.transformer.h)
    if hasattr(model,'model'):
        if hasattr(model.model,'layers'):
            return model.model.layers[layer],layer,len(model.model.layers)
    raise ValueError(f"Cannot find layers in {type(model)}")

def compute_grad_cov(model,tok,ds,layer,dev,n_cal=2000,ctx=32):
    blk,li,nl=get_layer_block(model,layer)
    d=model.config.hidden_size
    G=torch.zeros(d,d,dtype=torch.float64)
    cnt=0
    for seq in tqdm(ds[:n_cal],desc="grad cov"):
        x=seq[:ctx].unsqueeze(0).to(dev)
        y=seq[ctx].to(dev) if len(seq)>ctx else seq[-1].to(dev)
        cap=[None]
        def hk(m,i,o,c=cap):
            oo=o[0] if isinstance(o,tuple) else o
            c[0]=oo
        h=blk.register_forward_hook(hk)
        out=model(x)
        h.remove()
        hl=cap[0][0,-1,:].detach().float().requires_grad_(True)
        if hasattr(model,'transformer'):
            ln=model.transformer.ln_f
            lm=model.lm_head
        elif hasattr(model.model,'norm'):
            ln=model.model.norm
            lm=model.lm_head
        else:
            ln=torch.nn.Identity()
            lm=model.lm_head
        ln_w=ln.weight.data.clone() if hasattr(ln,'weight') else None
        ln_b=ln.bias.data.clone() if hasattr(ln,'bias') else None
        if ln_w is not None: ln.weight.data=ln_w.float()
        if ln_b is not None: ln.bias.data=ln_b.float()
        lm_w=lm.weight.data.clone(); lm.weight.data=lm_w.float()
        lm_b=None
        if hasattr(lm,'bias') and lm.bias is not None:
            lm_b=lm.bias.data.clone(); lm.bias.data=lm_b.float()
        logits=lm(ln(hl))
        loss=F.cross_entropy(logits.unsqueeze(0),y.unsqueeze(0))
        g=torch.autograd.grad(loss,hl)[0].detach().float().cpu()
        if ln_w is not None: ln.weight.data=ln_w
        if ln_b is not None: ln.bias.data=ln_b
        lm.weight.data=lm_w
        if lm_b is not None: lm.bias.data=lm_b
        G+=torch.outer(g.to(torch.float64),g.to(torch.float64))
        cnt+=1
        if cnt%500==0:
            gc.collect()
            if dev=="cuda": torch.cuda.empty_cache()
    G/=cnt
    evals,evecs=torch.linalg.eigh(G.float())
    idx=evals.argsort(descending=True)
    return evecs[:,idx],evals[idx]

def compute_margins(model,ds,layer,dev,evecs,n_eval=200,ctx=32,ks=[64,128,256]):
    blk,li,_=get_layer_block(model,layer)
    d=model.config.hidden_size
    hs=[]
    cap=[None]
    def hk(m,i,o,c=cap):
        oo=o[0] if isinstance(o,tuple) else o
        c[0]=oo.detach()
    handle=blk.register_forward_hook(hk)
    for seq in tqdm(ds[:n_eval],desc="embed margins"):
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad(): model(x)
        hs.append(cap[0][0,-1,:].float().cpu())
    handle.remove()
    H=torch.stack(hs)
    D_full=torch.cdist(H,H)
    D_full.fill_diagonal_(float('inf'))
    full_margins=D_full.min(dim=1).values
    results={"full_margin":float(full_margins.median().item())}
    for k in ks:
        U=evecs[:,:k]
        H_B=H@U@U.T
        H_I=H-H_B
        D_B=torch.cdist(H_B,H_B); D_B.fill_diagonal_(float('inf'))
        D_I=torch.cdist(H_I,H_I); D_I.fill_diagonal_(float('inf'))
        bm=D_B.min(dim=1).values
        im=D_I.min(dim=1).values
        results[f"behavior_margin_k{k}"]=float(bm.median().item())
        results[f"identity_margin_k{k}"]=float(im.median().item())
        results[f"behavior_frac_k{k}"]=float((bm.median()/full_margins.median()).item())
        results[f"identity_frac_k{k}"]=float((im.median()/full_margins.median()).item())
        torch.manual_seed(42)
        R=torch.randn(d,k)
        Q,_=torch.linalg.qr(R)
        H_R=H@Q@Q.T
        D_R=torch.cdist(H_R,H_R); D_R.fill_diagonal_(float('inf'))
        rm=D_R.min(dim=1).values
        results[f"random_frac_k{k}"]=float((rm.median()/full_margins.median()).item())
    return results

def compute_kl(model,ds,layer,dev,evecs,k=128,n_eval=500,ctx=32):
    blk,li,_=get_layer_block(model,layer)
    U=evecs[:,:k].to(dev)
    beh_kls,id_kls=[],[]
    cap=[None]
    def hk(m,i,o,c=cap):
        oo=o[0] if isinstance(o,tuple) else o
        c[0]=oo.detach()
    handle=blk.register_forward_hook(hk)
    for seq in tqdm(ds[:n_eval],desc="KL eval"):
        x=seq[:ctx].unsqueeze(0).to(dev)
        with torch.no_grad():
            out=model(x)
            bl=out.logits[0,-1,:]
            bp=F.softmax(bl.float(),dim=-1)
            h=cap[0][0,-1,:].float()
        h_B=(h@U@U.T)
        h_I=h-h_B
        for h_proj,store in [(h_B,beh_kls),(h_I,id_kls)]:
            def inject(m,i,o,hp=h_proj):
                oo=o[0] if isinstance(o,tuple) else o
                oo=oo.clone()
                oo[0,-1,:]=hp.to(oo.dtype)
                if isinstance(o,tuple): return (oo,)+o[1:]
                return oo
            hh=blk.register_forward_hook(inject)
            with torch.no_grad():
                out2=model(x)
            hh.remove()
            lp=out2.logits[0,-1,:].float()
            kl=F.kl_div(F.log_softmax(lp,dim=-1),bp,reduction='sum',log_target=False).item()
            store.append(kl)
    handle.remove()
    return float(np.median(beh_kls)),float(np.median(id_kls))

def run_model(mname,label,arch,params,d,nl,layer,dev):
    print(f"\n{'='*60}\n{label} ({mname})\nd={d} nl={nl} layer={layer}\n{'='*60}")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(mname,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    dtype=torch.float16 if d>1600 else torch.float32
    model=AutoModelForCausalLM.from_pretrained(mname,torch_dtype=dtype,output_hidden_states=True,trust_remote_code=True)
    model.eval().to(dev)
    for p in model.parameters(): p.requires_grad_(False)
    ds=make_ds(tok,n=2500,sl=33)
    evecs,evals=compute_grad_cov(model,tok,ds,layer,dev,n_cal=2000)
    energy=[]
    cs=evals.cumsum(0)/evals.sum()
    for kk in [32,64,128,256,512]:
        if kk<len(evals):
            energy.append({"k":kk,"energy_frac":float(cs[kk-1].item())})
    r95=0
    for i,c in enumerate(cs):
        if c>=0.95: r95=i+1; break
    margins=compute_margins(model,ds,layer,dev,evecs,n_eval=200,ks=[64,128,256])
    beh_kl,id_kl=compute_kl(model,ds,layer,dev,evecs,k=128,n_eval=500)
    result={
        "model":mname,"label":label,"architecture":arch,"params":params,
        "d":d,"n_layers":nl,"layer":layer,"k":128,"n_cal":2000,
        "r95":r95,"energy":energy,
        "utility":{"behavior":beh_kl,"identity":id_kl},
        **margins,
        "behavior_frac":margins.get("behavior_frac_k128",0),
        "identity_frac":margins.get("identity_frac_k128",0),
        "sqrt_k_over_d":math.sqrt(128/d),
        "sqrt_r95_over_d":math.sqrt(r95/d) if r95>0 else 0,
        "timestamp":time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    slug=mname.replace("/","_")
    with open(OUT/f"{slug}.json","w") as f:
        json.dump(result,f,indent=2)
    print(f"\nSaved: {slug}.json")
    print(f"  full_margin={margins['full_margin']:.3f}")
    print(f"  beh_frac={margins.get('behavior_frac_k128',0):.3f} id_frac={margins.get('identity_frac_k128',0):.3f}")
    print(f"  sqrt(k/d)={math.sqrt(128/d):.3f}")
    print(f"  beh_kl={beh_kl:.3f} id_kl={id_kl:.3f}")
    print(f"  r95={r95}")
    del model; gc.collect()
    if dev=="cuda": torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--models",nargs="+",default=None)
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    dev=args.device
    t0=time.time()
    targets=MODELS
    if args.models:
        targets=[m for m in MODELS if any(q in m[0] for q in args.models)]
    for mname,label,arch,params,d,nl,layer in targets:
        try:
            run_model(mname,label,arch,params,d,nl,layer,dev)
        except Exception as e:
            print(f"FAILED {mname}: {e}")
    print(f"\nAll done. Total: {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()

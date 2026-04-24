#!/usr/bin/env python3
import torch,json,numpy as np
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset

OUT=Path("artifacts/adversarial"); OUT.mkdir(parents=True,exist_ok=True)
DEV="mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEV}")

ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
texts=[t for t in ds["text"] if len(t.strip())>100]

def run_model(model_name,layers,ks,n_cal=1000,n_test=200,pfx=32):
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    tok=AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    mdl=AutoModelForCausalLM.from_pretrained(model_name).to(DEV).eval()
    d=mdl.config.hidden_size
    block_prefix="transformer.h" if hasattr(mdl,"transformer") else "model.layers"
    def get_block(l):
        if hasattr(mdl,"transformer"): return mdl.transformer.h[l]
        return mdl.model.layers[l]

    all_results={}
    for L in layers:
        print(f"\n--- Layer {L} ---")
        print("Computing gradient covariance...")
        C=torch.zeros(d,d,dtype=torch.float64)
        ct=0
        for i in range(min(len(texts),n_cal*3)):
            ids=tok(texts[i],return_tensors="pt",truncation=True,max_length=pfx+1)["input_ids"]
            if ids.shape[1]<pfx+1: continue
            ids=ids[:,:pfx+1].to(DEV)
            inp,lbl=ids[:,:pfx],ids[:,pfx]
            h_s=[None]
            def hk(m,i,o,s=h_s):
                s[0]=o[0] if isinstance(o,tuple) else o
                s[0].requires_grad_(True); s[0].retain_grad()
                return s[0] if not isinstance(o,tuple) else (s[0],)+o[1:]
            handle=get_block(L).register_forward_hook(hk)
            out=mdl(inp)
            loss=torch.nn.functional.cross_entropy(out.logits[0,-1].unsqueeze(0),lbl)
            loss.backward(); handle.remove()
            if h_s[0].grad is not None:
                g=h_s[0].grad[0,-1].detach().cpu().double()
                C+=g.unsqueeze(1)@g.unsqueeze(0); ct+=1
            mdl.zero_grad()
            if ct>=n_cal: break
        C/=ct
        evals,evecs=torch.linalg.eigh(C)
        idx=torch.argsort(evals,descending=True)
        evecs=evecs[:,idx].float().to(DEV)

        h_norms=[]
        def get_norm(m,i,o,s=h_norms):
            h=o[0] if isinstance(o,tuple) else o
            s.append(h[0,-1].detach().cpu().norm().item()); return o
        handle=get_block(L).register_forward_hook(get_norm)
        for i in range(min(50,len(texts)-n_cal*3)):
            ids=tok(texts[i+n_cal*3],return_tensors="pt",truncation=True,max_length=pfx)["input_ids"].to(DEV)
            if ids.shape[1]>=pfx:
                with torch.no_grad(): mdl(ids)
        handle.remove()
        h_norm=np.mean(h_norms)

        for K in ks:
            if K>=d: continue
            print(f"  k={K}...")
            U_B=evecs[:,:K]; P_B=U_B@U_B.T; P_I=torch.eye(d,device=DEV)-P_B
            sigmas=[0.5,1.0,2.0,5.0]
            res={s:{"identity_kl":[],"behavior_kl":[],"random_kl":[],
                     "identity_top1":[],"behavior_top1":[],"random_top1":[]} for s in sigmas}

            test_start=n_cal*3+50
            ti_done=0
            for ti in range(min(len(texts)-test_start,n_test*2)):
                ids=tok(texts[test_start+ti],return_tensors="pt",truncation=True,max_length=pfx)["input_ids"].to(DEV)
                if ids.shape[1]<pfx: continue

                bh=[None]
                def sh(m,i,o,s=bh):
                    s[0]=(o[0] if isinstance(o,tuple) else o)[0,-1].detach().clone(); return o
                handle=get_block(L).register_forward_hook(sh)
                with torch.no_grad(): base_out=mdl(ids)
                handle.remove()
                base_logits=base_out.logits[0,-1]
                base_probs=torch.softmax(base_logits,dim=-1)
                base_top1=base_logits.argmax().item()

                for sigma in sigmas:
                    noise_raw=torch.randn(d,device=DEV)
                    target_norm=sigma*h_norm
                    for mode,proj in [("identity",P_I),("behavior",P_B),("random",None)]:
                        noise=(proj@noise_raw) if proj is not None else noise_raw
                        n=noise.norm()
                        if n>0: noise=noise*(target_norm/n)
                        def inject(m,i,o,n=noise):
                            h=o[0] if isinstance(o,tuple) else o
                            h2=h.clone(); h2[0,-1]+=n
                            return (h2,)+o[1:] if isinstance(o,tuple) else h2
                        handle=get_block(L).register_forward_hook(inject)
                        with torch.no_grad(): p_out=mdl(ids)
                        handle.remove()
                        p_logits=p_out.logits[0,-1]
                        kl=torch.nn.functional.kl_div(torch.log_softmax(p_logits,dim=-1),base_probs,reduction='sum',log_target=False).item()
                        res[sigma][f"{mode}_kl"].append(kl)
                        res[sigma][f"{mode}_top1"].append(1 if p_logits.argmax().item()==base_top1 else 0)

                ti_done+=1
                if ti_done>=n_test: break

            key=f"layer{L}_k{K}"
            summary={}
            for sigma in sigmas:
                r=res[sigma]
                ik=np.median(r["identity_kl"]); bk=np.median(r["behavior_kl"]); rk=np.median(r["random_kl"])
                it=np.mean(r["identity_top1"]); bt=np.mean(r["behavior_top1"]); rt=np.mean(r["random_top1"])
                ratio=bk/max(ik,1e-10)
                summary[str(sigma)]={"id_kl":round(ik,4),"beh_kl":round(bk,4),"rnd_kl":round(rk,4),
                                     "id_top1":round(it,3),"beh_top1":round(bt,3),"rnd_top1":round(rt,3),
                                     "beh_id_ratio":round(ratio,2)}
                print(f"    σ={sigma}: id_kl={ik:.3f} beh_kl={bk:.3f} ratio={ratio:.1f}x id_top1={it:.1%} beh_top1={bt:.1%}")
            all_results[key]={"n_test":ti_done,"n_cal":ct,"h_norm":round(h_norm,1),"results":summary}

    return all_results

r1=run_model("openai-community/gpt2",layers=[6,11],ks=[64,128,256],n_test=200)
with open(OUT/"adversarial_gpt2small.json","w") as f: json.dump(r1,f,indent=2)

del torch.mps.current_allocated_memory
torch.mps.empty_cache() if hasattr(torch.mps,'empty_cache') else None

r2=run_model("openai-community/gpt2-medium",layers=[12],ks=[64,128,256],n_test=200)
with open(OUT/"adversarial_gpt2medium.json","w") as f: json.dump(r2,f,indent=2)

print("\n\nDone. Results saved to artifacts/adversarial/")

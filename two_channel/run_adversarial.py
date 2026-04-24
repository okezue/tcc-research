#!/usr/bin/env python3
import torch,json,numpy as np
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM

DEV="mps" if torch.backends.mps.is_available() else "cpu"
OUT=Path("artifacts/adversarial"); OUT.mkdir(parents=True,exist_ok=True)

print(f"Device: {DEV}")
tok=AutoTokenizer.from_pretrained("openai-community/gpt2")
if tok.pad_token is None: tok.pad_token=tok.eos_token
mdl=AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEV).eval()

L=6; d=768; N_cal=1000; N_test=100; K=128; PFX=32

print("Computing gradient covariance...")
from datasets import load_dataset
ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train",trust_remote_code=True)
texts=[t for t in ds["text"] if len(t.strip())>100]
C=torch.zeros(d,d,dtype=torch.float64)
ct=0
for i in range(min(len(texts),N_cal*2)):
    ids=tok(texts[i],return_tensors="pt",truncation=True,max_length=PFX+1)["input_ids"]
    if ids.shape[1]<PFX+1: continue
    ids=ids[:,:PFX+1].to(DEV)
    inp,lbl=ids[:,:PFX],ids[:,PFX]
    h_store=[None]
    def hook_fn(m,i,o,store=h_store):
        store[0]=o[0] if isinstance(o,tuple) else o
        store[0].requires_grad_(True)
        store[0].retain_grad()
        return store[0] if not isinstance(o,tuple) else (store[0],)+o[1:]
    handle=mdl.transformer.h[L].register_forward_hook(hook_fn)
    out=mdl(inp)
    logits=out.logits[0,-1]
    loss=torch.nn.functional.cross_entropy(logits.unsqueeze(0),lbl)
    loss.backward()
    handle.remove()
    if h_store[0].grad is not None:
        g=h_store[0].grad[0,-1].detach().cpu().double()
        C+=g.unsqueeze(1)@g.unsqueeze(0)
        ct+=1
    mdl.zero_grad()
    if ct>=N_cal: break
C/=ct
print(f"Computed gradient cov from {ct} samples")

evals,evecs=torch.linalg.eigh(C)
idx=torch.argsort(evals,descending=True)
evals=evals[idx]; evecs=evecs[:,idx]
U_B=evecs[:,:K].float().to(DEV)
P_B=(U_B@U_B.T)
P_I=torch.eye(d,device=DEV)-P_B

h_std_est=[]
def get_h_std(m,i,o,store=h_std_est):
    h=o[0] if isinstance(o,tuple) else o
    store.append(h[0,-1].detach().cpu().norm().item())
    return o
handle=mdl.transformer.h[L].register_forward_hook(get_h_std)
for i in range(50):
    ids=tok(texts[i+N_cal*2],return_tensors="pt",truncation=True,max_length=PFX)["input_ids"].to(DEV)
    if ids.shape[1]<PFX: continue
    with torch.no_grad(): mdl(ids)
handle.remove()
h_norm=np.mean(h_std_est)
print(f"Mean hidden state norm: {h_norm:.1f}")

sigmas=[0.1,0.5,1.0,2.0,5.0]
results={s:{"identity_kl":[],"behavior_kl":[],"random_kl":[],"identity_top1":[],"behavior_top1":[],"random_top1":[]} for s in sigmas}

test_texts=[t for t in texts[N_cal*2:] if len(t.strip())>100]
print(f"Running adversarial injection on {N_test} prefixes...")

for ti in range(N_test):
    ids=tok(test_texts[ti],return_tensors="pt",truncation=True,max_length=PFX)["input_ids"].to(DEV)
    if ids.shape[1]<PFX: continue

    baseline_h=[None]
    def store_h(m,i,o,s=baseline_h):
        s[0]=(o[0] if isinstance(o,tuple) else o)[0,-1].detach().clone()
        return o
    handle=mdl.transformer.h[L].register_forward_hook(store_h)
    with torch.no_grad():
        base_out=mdl(ids)
    handle.remove()
    base_logits=base_out.logits[0,-1]
    base_probs=torch.softmax(base_logits,dim=-1)
    base_top1=base_logits.argmax().item()
    h0=baseline_h[0]

    for sigma in sigmas:
        noise_raw=torch.randn(d,device=DEV)*sigma*(h_norm/np.sqrt(d))

        target_norm=sigma*(h_norm/np.sqrt(d))*np.sqrt(d)
        for mode,proj in [("identity",P_I),("behavior",P_B),("random",None)]:
            if proj is not None:
                noise=proj@noise_raw
            else:
                noise=noise_raw
            n=noise.norm()
            if n>0: noise=noise*(target_norm/n)

            def inject(m,i,o,n=noise):
                h=o[0] if isinstance(o,tuple) else o
                h_new=h.clone()
                h_new[0,-1]+=n
                return (h_new,)+o[1:] if isinstance(o,tuple) else h_new

            handle=mdl.transformer.h[L].register_forward_hook(inject)
            with torch.no_grad():
                perturbed_out=mdl(ids)
            handle.remove()

            p_logits=perturbed_out.logits[0,-1]
            p_probs=torch.softmax(p_logits,dim=-1)
            p_top1=p_logits.argmax().item()

            kl=torch.nn.functional.kl_div(
                torch.log_softmax(p_logits,dim=-1),
                base_probs,reduction='sum',log_target=False
            ).item()

            results[sigma][f"{mode}_kl"].append(kl)
            results[sigma][f"{mode}_top1"].append(1 if p_top1==base_top1 else 0)

    if (ti+1)%20==0:
        print(f"  {ti+1}/{N_test} done")

print("\n=== RESULTS ===")
print(f"{'sigma':>8} | {'Id KL':>10} {'Beh KL':>10} {'Rnd KL':>10} | {'Id top1':>8} {'Beh top1':>8} {'Rnd top1':>8}")
print("-"*80)
summary={}
for sigma in sigmas:
    r=results[sigma]
    ik=np.median(r["identity_kl"]); bk=np.median(r["behavior_kl"]); rk=np.median(r["random_kl"])
    it=np.mean(r["identity_top1"]); bt=np.mean(r["behavior_top1"]); rt=np.mean(r["random_top1"])
    print(f"{sigma:>8.1f} | {ik:>10.4f} {bk:>10.4f} {rk:>10.4f} | {it:>8.1%} {bt:>8.1%} {rt:>8.1%}")
    summary[str(sigma)]={
        "identity_kl_median":ik,"behavior_kl_median":bk,"random_kl_median":rk,
        "identity_top1_rate":it,"behavior_top1_rate":bt,"random_top1_rate":rt,
        "identity_kl_all":r["identity_kl"],"behavior_kl_all":r["behavior_kl"],
        "random_kl_all":r["random_kl"],
        "beh_id_kl_ratio":bk/max(ik,1e-10)
    }

with open(OUT/"adversarial_results.json","w") as f:
    json.dump({"k":K,"layer":L,"n_test":N_test,"n_cal":ct,"sigmas":sigmas,"results":summary},f,indent=2)
print(f"\nSaved to {OUT}/adversarial_results.json")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import json
import argparse

def load_model(model_id: str, device: str="cpu"):
    tok=AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(model_id,output_hidden_states=True)
    m.eval()
    m.to(device)
    return m,tok

def get_layer_block(model, layer_idx: int):
    if hasattr(model,'transformer'):
        blocks=model.transformer.h
    elif hasattr(model,'model') and hasattr(model.model,'layers'):
        blocks=model.model.layers
    else:
        raise ValueError("Unknown model architecture")
    if layer_idx<0:
        layer_idx=len(blocks)+layer_idx
    return blocks[layer_idx], layer_idx, len(blocks)

def compute_gradient_covariance(
    model, tokenizer, layer_idx: int,
    dataset: list, prefix_len: int=32,
    device: str="cpu", max_samples: int=10000
):
    block, abs_idx, n_layers=get_layer_block(model,layer_idx)

    d=model.config.hidden_size
    C=torch.zeros(d,d,dtype=torch.float64)
    act_sum=torch.zeros(d,dtype=torch.float64)
    act_sq=torch.zeros(d,d,dtype=torch.float64)
    n=0

    captured=[None]
    def hook_fn(module,inp,out):
        o=out[0] if isinstance(out,tuple) else out
        o.retain_grad()
        captured[0]=o
    handle=block.register_forward_hook(hook_fn)

    for i in tqdm(range(min(len(dataset),max_samples)),desc=f"Grad cov layer {abs_idx}"):
        toks=dataset[i]
        if len(toks)<prefix_len+1:
            continue
        x=toks[:prefix_len].unsqueeze(0).to(device)
        y=toks[prefix_len].unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)
        captured[0]=None
        out=model(x)
        h_full=captured[0]
        if h_full is None or h_full.grad_fn is None:
            continue
        h=h_full[0,-1,:]

        logits=out.logits[0,-1,:]
        loss=F.cross_entropy(logits.unsqueeze(0),y)
        loss.backward()

        g_full=h_full.grad
        if g_full is None:
            continue
        g=g_full[0,-1,:].detach().cpu().to(torch.float64)
        C+=torch.outer(g,g)

        hc=h.detach().cpu().to(torch.float64)
        act_sum+=hc
        act_sq+=torch.outer(hc,hc)
        n+=1

    handle.remove()

    if n==0:
        raise RuntimeError("No valid samples")

    C/=n
    act_mean=act_sum/n
    act_cov=act_sq/n-torch.outer(act_mean,act_mean)

    evals_grad, evecs_grad=torch.linalg.eigh(C)
    evals_grad=evals_grad.flip(0)
    evecs_grad=evecs_grad.flip(1)

    evals_act, evecs_act=torch.linalg.eigh(act_cov)
    evals_act=evals_act.flip(0)
    evecs_act=evecs_act.flip(1)

    return {
        "grad_eigenvalues": evals_grad,
        "grad_eigenvectors": evecs_grad,
        "act_eigenvalues": evals_act,
        "act_eigenvectors": evecs_act,
        "n_samples": n,
        "layer_idx": abs_idx,
    }

def save_subspace(results: dict, out_dir: Path, model_id: str, layer_idx: int):
    d=out_dir/model_id.replace("/","_")/f"layer_{layer_idx}"
    d.mkdir(parents=True,exist_ok=True)
    torch.save(results["grad_eigenvectors"],d/"grad_evecs.pt")
    torch.save(results["grad_eigenvalues"],d/"grad_evals.pt")
    torch.save(results["act_eigenvectors"],d/"act_evecs.pt")
    torch.save(results["act_eigenvalues"],d/"act_evals.pt")
    with open(d/"info.json","w") as f:
        json.dump({"n_samples":results["n_samples"],"layer_idx":results["layer_idx"]},f)
    print(f"Saved to {d}")
    return d

def load_subspace(path: Path, k: int, mode: str="grad"):
    evecs=torch.load(path/f"{mode}_evecs.pt",weights_only=True)
    evals=torch.load(path/f"{mode}_evals.pt",weights_only=True)
    U=evecs[:,:k].to(torch.float32)
    return U, evals[:k].to(torch.float32)

def generate_random_subspace(d: int, k: int, seed: int=42) -> torch.Tensor:
    g=torch.Generator()
    g.manual_seed(seed)
    M=torch.randn(d,k,generator=g)
    Q,_=torch.linalg.qr(M)
    return Q

def make_calibration_dataset(tokenizer, n: int=10000, seq_len: int=33, seed: int=42):
    from datasets import load_dataset
    ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    torch.manual_seed(seed)
    all_toks=[]
    for row in ds:
        txt=row["text"].strip()
        if len(txt)<50:
            continue
        ids=tokenizer(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=seq_len:
            start=torch.randint(0,len(ids)-seq_len+1,(1,)).item()
            all_toks.append(torch.tensor(ids[start:start+seq_len],dtype=torch.long))
            if len(all_toks)>=n:
                break
    print(f"Calibration dataset: {len(all_toks)} sequences of length {seq_len}")
    return all_toks

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model-id",default="openai-community/gpt2")
    p.add_argument("--layers",nargs="+",type=int,default=[6,-1])
    p.add_argument("--n-samples",type=int,default=5000)
    p.add_argument("--prefix-len",type=int,default=32)
    p.add_argument("--device",default="mps")
    p.add_argument("--out-dir",default="artifacts/subspace")
    args=p.parse_args()

    model,tok=load_model(args.model_id,args.device)
    ds=make_calibration_dataset(tok,n=args.n_samples,seq_len=args.prefix_len+1)

    for li in args.layers:
        res=compute_gradient_covariance(
            model,tok,li,ds,
            prefix_len=args.prefix_len,
            device=args.device,
            max_samples=args.n_samples
        )
        save_subspace(res,Path(args.out_dir),args.model_id,res["layer_idx"])

    print("Done computing subspaces.")

import argparse,json,os,glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from .split_memory_transformer import SMT
from .exp_smt_train import GPTBaseline

def make_ds(tok,n,sl,seed=42,corpus="wikitext"):
    from datasets import load_dataset
    if corpus=="wikitext":
        ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    elif corpus=="tinystories":
        ds=load_dataset("roneneldan/TinyStories",split="train")
    torch.manual_seed(seed)
    out=[]
    for row in ds:
        txt=row.get("text","").strip()
        if len(txt)<80:continue
        ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=sl:
            s=torch.randint(0,len(ids)-sl+1,(1,)).item()
            out.append(torch.tensor(ids[s:s+sl],dtype=torch.long))
            if len(out)>=n:break
    return out

def hidden_at_layer_smt(model,ids,layer):
    with torch.no_grad():
        logits,u_l,v_l=model(ids,return_uv=True)
    h=torch.cat([u_l[layer],v_l[layer]],dim=-1)
    return h

def hidden_at_layer_baseline(model,ids,layer):
    cap=[None]
    blk=model.layers[layer]
    def hk(m,i,o,c=cap):c[0]=o.detach()
    h=blk.register_forward_hook(hk)
    with torch.no_grad():model(ids)
    h.remove()
    return cap[0]

def compute_F_diag(model,ids_list,layer,is_smt,n_cal=300):
    d=None
    F_diag=None
    cnt=0
    for ids in tqdm(ids_list[:n_cal],desc="F_diag"):
        ids=ids.unsqueeze(0).to(next(model.parameters()).device)
        if is_smt:
            logits,u_l,v_l=model(ids,return_uv=True)
            h=torch.cat([u_l[layer],v_l[layer]],dim=-1)
        else:
            cap=[None]
            blk=model.layers[layer]
            def hk(m,i,o,c=cap):c[0]=o
            hh=blk.register_forward_hook(hk)
            logits=model(ids)
            hh.remove()
            h=cap[0]
        if d is None:
            d=h.shape[-1]
            F_diag=torch.zeros(d,dtype=torch.float64,device=h.device)
        h_var=h[0,-1,:].clone().detach().requires_grad_(True)
        if is_smt:
            r=u_l[0].shape[-1]
            u_inj=h_var[:r].unsqueeze(0).unsqueeze(0)
            v_inj=h_var[r:].unsqueeze(0).unsqueeze(0)
            u_l[layer][:,-1:,:r if False else u_l[layer].size(-1)]=0
        h_var2=h[:,-1,:].clone()
        h_var2[0]=h_var
        last_logits=logits[0,-1,:].float()
        p=F.softmax(last_logits,dim=-1).detach()
        topv=torch.topk(p,5).indices.tolist()
        gd=torch.zeros(d,dtype=torch.float64)
        for v in topv:
            try:
                g=torch.autograd.grad(F.log_softmax(last_logits,dim=-1)[v],h_var2,retain_graph=True)[0]
                if g is None:continue
                gv=g[0].float().detach().cpu().double()
                gd+=p[v].item()*gv.pow(2)
            except RuntimeError:
                continue
        F_diag+=gd.to(F_diag.device)
        cnt+=1
    return (F_diag/max(cnt,1)).float()

def compute_S_diag(model,ids_list,layer,is_smt,n_pairs=500):
    H=[]
    dev=next(model.parameters()).device
    for ids in tqdm(ids_list[:n_pairs*2],desc="hidden"):
        ids=ids.unsqueeze(0).to(dev)
        if is_smt:
            with torch.no_grad():
                _,u_l,v_l=model(ids,return_uv=True)
            h=torch.cat([u_l[layer],v_l[layer]],dim=-1)
        else:
            cap=[None]
            blk=model.layers[layer]
            def hk(m,i,o,c=cap):c[0]=o.detach()
            hh=blk.register_forward_hook(hk)
            with torch.no_grad():model(ids)
            hh.remove()
            h=cap[0]
        H.append(h[0,-1,:].float().cpu())
    H=torch.stack(H)
    n=H.shape[0]
    pairs_i=torch.randint(0,n,(n_pairs,))
    pairs_j=torch.randint(0,n,(n_pairs,))
    deltas=H[pairs_i]-H[pairs_j]
    S_diag=(deltas*deltas).mean(dim=0)
    return S_diag,H

def G_metrics(F_diag,S_diag,k=128):
    F_diag=F_diag.clamp(min=1e-8).double()
    S_diag=S_diag.clamp(min=1e-8).double()
    rho_F=F_diag/F_diag.sum()
    rho_S=S_diag/S_diag.sum()
    G_Mah_classical=F_diag.sum()*S_diag.sum()/(F_diag.sqrt()*S_diag.sqrt()).sum().pow(2)
    G_Mah=1.0/((rho_F.sqrt()*rho_S.sqrt()).sum()).pow(2).item()
    sorted_idx=torch.argsort(F_diag,descending=True)[:k]
    E_k=rho_F[sorted_idx].sum().item()
    q_B=rho_S[sorted_idx].sum().item()
    G_Mah_lower=1.0/(((E_k*q_B)**0.5)+(((1-E_k)*(1-q_B))**0.5))**2
    return dict(G_Mah=float(G_Mah),G_Mah_classical=float(G_Mah_classical),E_k=float(E_k),q_B=float(q_B),G_Mah_lower=float(G_Mah_lower),tr_F=float(F_diag.sum()),tr_S=float(S_diag.sum()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt",required=True)
    ap.add_argument("--info_json",required=True)
    ap.add_argument("--corpus",default="wikitext")
    ap.add_argument("--probe_layers",type=str,default="4,6,8")
    ap.add_argument("--n_cal",type=int,default=300)
    ap.add_argument("--n_pairs",type=int,default=500)
    ap.add_argument("--ctx",type=int,default=64)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    info=json.load(open(a.info_json))
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tok=AutoTokenizer.from_pretrained("openai-community/gpt2")
    if info["arch"]=="smt":
        model=SMT(vocab=tok.vocab_size,r=info["r"],m=info["m"],n_layers=info["n_layers"]).to(dev)
        is_smt=True
    else:
        d_total=info["r"]+info["m"]
        model=GPTBaseline(vocab=tok.vocab_size,d=d_total,n_layers=info["n_layers"],n_heads=8,ff=4*d_total).to(dev)
        is_smt=False
    model.load_state_dict(torch.load(a.ckpt,map_location=dev))
    model.eval()
    print(f"loaded {info['slug']}")
    print("loading data")
    ds=make_ds(tok,a.n_cal+a.n_pairs*2,a.ctx,corpus=a.corpus)
    cal_ids=ds[:a.n_cal]
    bank_ids=ds[a.n_cal:a.n_cal+a.n_pairs*2]
    layers=[int(x) for x in a.probe_layers.split(",")]
    results={}
    for L in layers:
        print(f"\n=== layer {L} ===")
        F_diag=compute_F_diag(model,cal_ids,L,is_smt,n_cal=a.n_cal)
        S_diag,_=compute_S_diag(model,bank_ids,L,is_smt,n_pairs=a.n_pairs)
        m=G_metrics(F_diag,S_diag,k=128)
        m["layer"]=L
        results[L]=m
        print(json.dumps(m,indent=2))
    out=dict(info=info,probe_results=results)
    os.makedirs(os.path.dirname(a.out)or".",exist_ok=True)
    with open(a.out,"w") as f:json.dump(out,f,indent=2)
    print(f"\nwrote {a.out}")

if __name__=="__main__":
    main()

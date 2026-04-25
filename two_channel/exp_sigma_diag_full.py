import argparse,json,os,time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from .adjacency_builder_v2 import get_layer_block,build_full_adjacency

def make_ds(tok,n,sl,seed=42):
    from datasets import load_dataset
    ds=load_dataset("wikitext","wikitext-103-raw-v1",split="train")
    torch.manual_seed(seed)
    out=[]
    for row in ds:
        txt=row["text"].strip()
        if len(txt)<80:continue
        ids=tok(txt,add_special_tokens=False,truncation=True,max_length=512)["input_ids"]
        if len(ids)>=sl:
            s=torch.randint(0,len(ids)-sl+1,(1,)).item()
            out.append(torch.tensor(ids[s:s+sl],dtype=torch.long))
            if len(out)>=n:break
    return out

def compute_F_diag(model,blk,prefixes,device,n_cal=300):
    d=model.config.hidden_size
    F_diag=torch.zeros(d,dtype=torch.float64)
    cnt=0
    for p in tqdm(prefixes[:n_cal],desc="F_diag"):
        ids=p.unsqueeze(0).to(device)
        cap=[None]
        def hkc(m,i,o,c=cap):
            c[0]=(o[0] if isinstance(o,tuple) else o).detach()
        h1=blk.register_forward_hook(hkc)
        with torch.no_grad():model(ids)
        h1.remove()
        h_orig=cap[0][0,-1,:].float()
        h_var=h_orig.clone().requires_grad_(True)
        def inj(m,i,o,hv=h_var):
            oo=o[0] if isinstance(o,tuple) else o
            oo=oo.clone()
            oo[0,-1,:]=hv.to(oo.dtype)
            if isinstance(o,tuple):return(oo,)+o[1:]
            return oo
        h2=blk.register_forward_hook(inj)
        out=model(ids)
        h2.remove()
        logits=out.logits[0,-1,:].float()
        p_v=F.softmax(logits,dim=-1).detach()
        lp=F.log_softmax(logits,dim=-1)
        topv=torch.topk(p_v,10).indices.tolist()
        gd=torch.zeros(d,dtype=torch.float64)
        for v in topv:
            g=torch.autograd.grad(lp[v],h_var,retain_graph=True)[0].detach().float().cpu().double()
            gd+=p_v[v].item()*g.pow(2)
        F_diag+=gd
        cnt+=1
    return (F_diag/cnt).float()

def embed_bank(model,blk,prefixes,device):
    H=[]
    cap=[None]
    def hk(m,i,o,c=cap):
        c[0]=(o[0] if isinstance(o,tuple) else o).detach()
    h=blk.register_forward_hook(hk)
    for p in tqdm(prefixes,desc="bank"):
        ids=p.unsqueeze(0).to(device)
        with torch.no_grad():model(ids)
        H.append(cap[0][0,-1,:].float().cpu())
    h.remove()
    return torch.stack(H)

def build_diag_sigma(F_diag,alpha,kappa):
    F_eff=F_diag.clamp(min=1e-8)
    w=F_eff.pow(-alpha)
    s=w*(2*kappa/(F_eff*w).sum())
    return s

def worst_mahal_diag(deltas,s):
    inv=1.0/s
    q=(deltas*deltas)@inv
    return q.max().item(),q.mean().item(),float(q.quantile(0.95).item())

def mahal_retrieval_top1(H_query_clean,H_bank_full,gt_idx,s,sigma_scale,seed=0,bsz=128):
    torch.manual_seed(seed)
    g=torch.Generator().manual_seed(seed)
    noise=torch.randn(H_query_clean.shape,generator=g)*s.sqrt()*sigma_scale
    Hq_noisy=H_query_clean+noise
    inv=1.0/s
    correct=0
    N=H_query_clean.shape[0]
    H_bank_dev=H_bank_full.cuda() if torch.cuda.is_available() else H_bank_full
    Hq_dev=Hq_noisy.cuda() if torch.cuda.is_available() else Hq_noisy
    inv_dev=inv.cuda() if torch.cuda.is_available() else inv
    gt_dev=gt_idx.cuda() if torch.cuda.is_available() else gt_idx
    for i in range(0,N,bsz):
        q=Hq_dev[i:i+bsz]
        diff=q.unsqueeze(1)-H_bank_dev.unsqueeze(0)
        dist=(diff*diff*inv_dev).sum(-1)
        idx=dist.argmin(dim=-1)
        correct+=(idx==gt_dev[i:i+bsz]).sum().item()
    return correct/N

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",required=True)
    ap.add_argument("--layers",type=str,required=True,help="comma-separated layer indices")
    ap.add_argument("--n_cal",type=int,default=300)
    ap.add_argument("--n_bank",type=int,default=50000)
    ap.add_argument("--n_query",type=int,default=2000)
    ap.add_argument("--n_each_adj",type=int,default=5000)
    ap.add_argument("--ctx",type=int,default=32)
    ap.add_argument("--alphas",type=str,default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--kappas",type=str,default="0.3,1,3,7")
    ap.add_argument("--sigma_scale",type=float,default=1.0,help="multiplicative noise scale on sqrt(s)")
    ap.add_argument("--out_dir",default="artifacts/sigma_diag_validate")
    ap.add_argument("--dtype",default="bfloat16")
    a=ap.parse_args()
    os.makedirs(a.out_dir,exist_ok=True)
    dev="cuda" if torch.cuda.is_available() else "cpu"
    dtype=getattr(torch,a.dtype)
    print(f"loading {a.model} on {dev} {a.dtype}")
    tok=AutoTokenizer.from_pretrained(a.model)
    model=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=dtype).to(dev).eval()
    print("building dataset")
    n_total=a.n_bank+a.n_query+a.n_cal+a.n_each_adj
    ds=make_ds(tok,n_total,a.ctx)
    cal_pref=ds[:a.n_cal]
    bank_pref=ds[a.n_cal:a.n_cal+a.n_bank]
    query_pref=ds[a.n_cal+a.n_bank:a.n_cal+a.n_bank+a.n_query]
    adj_pref=ds[-a.n_each_adj:]
    layers=[int(x) for x in a.layers.split(",")]
    alphas=[float(x) for x in a.alphas.split(",")]
    kappas=[float(x) for x in a.kappas.split(",")]
    for L in layers:
        t0=time.time()
        print(f"\n=== layer {L} ===")
        blk=get_layer_block(model,L)
        F_diag=compute_F_diag(model,blk,cal_pref,dev,n_cal=a.n_cal)
        print(f"  tr(F_diag)={F_diag.sum():.4f}")
        H_bank_distract=embed_bank(model,blk,bank_pref,dev)
        H_query=embed_bank(model,blk,query_pref,dev)
        H_bank=torch.cat([H_query,H_bank_distract[:max(0,a.n_bank-a.n_query)]],dim=0)
        gt_idx=torch.arange(a.n_query,dtype=torch.long)
        print(f"  building adjacency 4x{a.n_each_adj}")
        deltas,sizes=build_full_adjacency(model,tok,blk,adj_pref,dev,n_each=a.n_each_adj,k_alt=256)
        print(f"  adjacency: {deltas.shape}, sizes={sizes}")
        rows=[]
        for al in alphas:
            for kp in kappas:
                s=build_diag_sigma(F_diag,al,kp)
                wm,mn,p95=worst_mahal_diag(deltas,s)
                top1=mahal_retrieval_top1(H_query,H_bank,gt_idx,s,a.sigma_scale)
                rows.append(dict(alpha=al,kappa=kp,worst_mahal=wm,mean_mahal=mn,p95_mahal=p95,retrieval_top1=top1,trF_s=(F_diag*s).sum().item()))
                print(f"  α={al:.2f} κ={kp:.2f} worst={wm:.2f} top1={top1:.3f}")
        out=dict(model=a.model,layer=L,d=int(F_diag.numel()),n_cal=a.n_cal,n_bank=a.n_bank,n_query=a.n_query,n_each_adj=a.n_each_adj,adj_sizes=sizes,F_diag_sum=float(F_diag.sum()),F_diag_max=float(F_diag.max()),sigma_scale=a.sigma_scale,rows=rows,elapsed_s=time.time()-t0)
        slug=a.model.replace("/","_")
        with open(os.path.join(a.out_dir,f"sigma_diag_{slug}_L{L}.json"),"w") as f:
            json.dump(out,f,indent=2)
        torch.save(F_diag,os.path.join(a.out_dir,f"F_diag_{slug}_L{L}.pt"))
        print(f"  saved layer {L} in {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()

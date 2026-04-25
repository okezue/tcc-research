import argparse,json,os,glob
import torch
import numpy as np
def build_diag_sigma(F_diag,alpha,kappa):
    w=F_diag.pow(-alpha)
    s=w*(2*kappa/(F_diag*w).sum())
    return s
def worst_mahal(deltas,s):
    inv=1.0/s
    q=(deltas*deltas)@inv
    return q.max().item(),q.mean().item(),float(np.percentile(q.cpu().numpy(),95))
def load_F_and_deltas(model_dir,layer):
    Fp=os.path.join(model_dir,f"F_diag_layer{layer}.pt")
    Dp=os.path.join(model_dir,f"deltas_layer{layer}.pt")
    if not(os.path.exists(Fp) and os.path.exists(Dp)):
        return None,None
    return torch.load(Fp),torch.load(Dp)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_dir",required=True)
    ap.add_argument("--layer",type=int,required=True)
    ap.add_argument("--alphas",type=str,default="0,0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--kappas",type=str,default="0.3,1,3,7")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    F,D=load_F_and_deltas(a.model_dir,a.layer)
    if F is None:
        raise FileNotFoundError("missing F_diag or deltas tensors")
    F=F.float().clamp_min(1e-8)
    D=D.float()
    rows=[]
    for al in [float(x) for x in a.alphas.split(",")]:
        for kp in [float(x) for x in a.kappas.split(",")]:
            s=build_diag_sigma(F,al,kp)
            mx,mn,p95=worst_mahal(D,s)
            rows.append(dict(alpha=al,kappa=kp,worst=mx,mean=mn,p95=p95))
    out=dict(model_dir=a.model_dir,layer=a.layer,d=int(F.numel()),n_deltas=int(D.shape[0]),rows=rows)
    os.makedirs(os.path.dirname(a.out)or".",exist_ok=True)
    with open(a.out,"w")as f:json.dump(out,f,indent=2)
    print(f"wrote {a.out}")
if __name__=="__main__":
    main()

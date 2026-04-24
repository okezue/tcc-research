#!/usr/bin/env python3
"""Train the learned inverter on (released hidden state -> prefix tokens) pairs.

Setting: clean last-token hidden state h at layer L, reconstruct the 32-token
prefix that produced it. Inversion quality measured by exact-sequence match,
token accuracy, and edit distance. Supports corruption-aware training where
h is perturbed by the defender's mechanism before being fed to the inverter.
"""
import os,sys,json,time,argparse,math
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from two_channel.learned_inverter import InverterDecoder
from two_channel.exp_optimal_defense import make_ds,get_layer_block,compute_fisher_avg,compute_id_cov,embed_bank,gen_eigendecomp
from two_channel.mahalanobis_defense import solve_mahalanobis_optimal

DEV="cuda" if torch.cuda.is_available() else "cpu"
OUT=Path("artifacts/learned_inverter")
OUT.mkdir(parents=True,exist_ok=True)

def build_pair_tensors(model,tok,ds,layer,dev,ctx=32,bs=16):
    blk,_,_=get_layer_block(model,layer)
    cap=[None]
    def hk(m,i,o,c=cap):
        c[0]=(o[0] if isinstance(o,tuple) else o).detach()
    h1=blk.register_forward_hook(hk)
    H=[];X=[]
    i=0
    for i in tqdm(range(0,len(ds),bs),desc="embed"):
        batch=[s[:ctx] for s in ds[i:i+bs]]
        if not batch: continue
        x=torch.stack(batch).to(dev)
        with torch.no_grad(): model(x)
        H.append(cap[0][:,-1,:].float().cpu())
        X.append(x.cpu())
    h1.remove()
    return torch.cat(H,0),torch.cat(X,0)

def corrupt(H,mech,sigma,d,Sigma_mah=None,V_gen=None,U_B=None,k=128):
    g=torch.randn_like(H)
    if mech=="clean": return H
    if mech=="iso": return H+g*sigma
    if mech=="complement":
        z=g*sigma
        return H+(z-z@U_B@U_B.T)
    if mech=="gen_eigen":
        z=torch.zeros(H.shape[0],d)
        z[:,:k]=torch.randn(H.shape[0],k)*sigma
        return H+z@V_gen.T
    if mech=="mah":
        ev,U=torch.linalg.eigh((Sigma_mah+Sigma_mah.T)/2)
        L=U@torch.diag(ev.clamp(min=0).sqrt())
        z=torch.randn(H.shape[0],d)
        return H+z@L.T
    raise ValueError(mech)

def exact_match(pred,target):
    return (pred==target).all(-1).float().mean().item()

def tok_acc(pred,target):
    return (pred==target).float().mean().item()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="openai-community/gpt2")
    p.add_argument("--layer",type=int,default=6)
    p.add_argument("--n_train",type=int,default=50000)
    p.add_argument("--n_test",type=int,default=2000)
    p.add_argument("--ctx",type=int,default=32)
    p.add_argument("--steps",type=int,default=15000)
    p.add_argument("--bs",type=int,default=64)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--d_dec",type=int,default=512)
    p.add_argument("--layers",type=int,default=6)
    p.add_argument("--heads",type=int,default=8)
    p.add_argument("--M",type=int,default=16)
    p.add_argument("--train_corrupt",default="clean")
    p.add_argument("--train_sigma",type=float,default=1.0)
    p.add_argument("--eval_sigmas",default="1.0,3.0,5.0")
    p.add_argument("--device",default=DEV)
    args=p.parse_args()
    dev=args.device
    t0=time.time()
    print(f"[inverter] {args.model} L{args.layer} steps={args.steps} bs={args.bs}")

    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    is_gpt2="gpt2" in args.model.lower()
    dtype=torch.float32 if is_gpt2 else torch.float16
    lm=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=dtype,output_hidden_states=True,trust_remote_code=True)
    lm.eval().to(dev)
    for pp in lm.parameters(): pp.requires_grad_(False)
    d=lm.config.hidden_size
    V=lm.config.vocab_size

    n_need=args.n_train+args.n_test+500
    print(f"  building {n_need} prefixes...")
    ds=make_ds(tok,n=n_need,sl=args.ctx+1)

    print(f"  computing F, S, gen-eigen, Sigma_mah for corruption menu...")
    F_mat=compute_fisher_avg(lm,ds[:300],args.layer,dev,n_cal=300,ctx=args.ctx)
    H_cal,_=build_pair_tensors(lm,tok,ds[300:1000],args.layer,dev,ctx=args.ctx,bs=32)
    S=compute_id_cov(H_cal,n_pairs=min(len(H_cal),1500))
    lambdas,V_gen,_,_=gen_eigendecomp(S,F_mat)
    V_gen_norm=V_gen.clone()
    for i in range(V_gen_norm.shape[1]):
        fv=(V_gen_norm[:,i].double()@F_mat.double()@V_gen_norm[:,i].double()).sqrt().clamp(min=1e-8)
        V_gen_norm[:,i]=V_gen_norm[:,i]/fv.float()
    cov=H_cal.T@H_cal/H_cal.shape[0]
    evals_cov,evecs_cov=torch.linalg.eigh(cov)
    idx_cov=evals_cov.argsort(descending=True)
    U_B=evecs_cov[:,idx_cov][:,:128]
    kappa_trial=1.0*1.0*F_mat.trace().item()
    mh=solve_mahalanobis_optimal(F_mat,S,kappa_trial,eta_ratio=1e-3)
    Sigma_mah_ref=mh["Sigma_star"]

    print(f"  embedding train set ({args.n_train})...")
    H_tr,X_tr=build_pair_tensors(lm,tok,ds[:args.n_train],args.layer,dev,ctx=args.ctx,bs=32)
    print(f"  embedding test set ({args.n_test})...")
    H_te,X_te=build_pair_tensors(lm,tok,ds[args.n_train:args.n_train+args.n_test],args.layer,dev,ctx=args.ctx,bs=32)
    print(f"  H_tr={H_tr.shape} X_tr={X_tr.shape}")

    inv=InverterDecoder(d_in=d,vocab=V,d=args.d_dec,heads=args.heads,layers=args.layers,M=args.M,T_max=args.ctx+1).to(dev)
    n_params=sum(p.numel() for p in inv.parameters())
    print(f"  inverter params: {n_params/1e6:.1f}M")
    opt=torch.optim.AdamW(inv.parameters(),lr=args.lr,weight_decay=0.01,betas=(0.9,0.95))
    warmup=min(500,args.steps//20)
    def lr_sched(step):
        if step<warmup: return step/warmup
        t=(step-warmup)/max(1,args.steps-warmup)
        return 0.5*(1+math.cos(math.pi*t))

    log=[]
    bos=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    for step in range(args.steps):
        idx=torch.randint(0,len(H_tr),(args.bs,))
        h=H_tr[idx].to(dev)
        x=X_tr[idx].to(dev)
        if args.train_corrupt!="clean":
            h_cpu=h.cpu()
            h_cpu=corrupt(h_cpu,args.train_corrupt,args.train_sigma,d,Sigma_mah=Sigma_mah_ref,V_gen=V_gen_norm,U_B=U_B)
            h=h_cpu.to(dev)
        x_bos=torch.cat([torch.full((args.bs,1),bos,dtype=torch.long,device=dev),x],dim=1)
        loss=inv.loss(h,x_bos)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inv.parameters(),1.0)
        for g in opt.param_groups: g["lr"]=args.lr*lr_sched(step)
        opt.step()
        if step%200==0 or step==args.steps-1:
            print(f"  step {step}  loss={loss.item():.3f}  lr={opt.param_groups[0]['lr']:.2e}")
            log.append({"step":step,"loss":loss.item()})

    print(f"  evaluating on {args.n_test} test prefixes across corruption menu...")
    eval_rows=[]
    T=args.ctx
    with torch.no_grad():
        for mech in ["clean","iso","complement","gen_eigen","mah"]:
            for sigma in ([0.0] if mech=="clean" else [float(x) for x in args.eval_sigmas.split(",")]):
                if mech=="clean" and sigma!=0: continue
                H_eval=corrupt(H_te.clone(),mech,sigma,d,Sigma_mah=Sigma_mah_ref,V_gen=V_gen_norm,U_B=U_B)
                em=0;ta=0;n_batch=0
                bs=64
                for i in range(0,len(H_eval),bs):
                    hh=H_eval[i:i+bs].to(dev)
                    xx=X_te[i:i+bs].to(dev)
                    pred=inv.greedy(hh,bos=bos,T=T+1)
                    pred=pred[:,1:]
                    em+=(pred==xx).all(-1).float().sum().item()
                    ta+=(pred==xx).float().mean(-1).sum().item()
                    n_batch+=len(hh)
                em/=n_batch;ta/=n_batch
                eval_rows.append({"mech":mech,"sigma":sigma,"exact_match":em,"tok_acc":ta})
                print(f"    {mech:12s} sig={sigma} EM={em:.3f} TA={ta:.3f}")

    slug=args.model.replace("/","_")
    out={
        "model":args.model,"layer":args.layer,"d":d,"ctx":args.ctx,
        "n_train":args.n_train,"n_test":args.n_test,"steps":args.steps,
        "train_corrupt":args.train_corrupt,"train_sigma":args.train_sigma,
        "n_params":int(n_params),
        "train_log":log,"eval":eval_rows,
        "elapsed_s":time.time()-t0,
    }
    with open(OUT/f"learned_inverter_{slug}_{args.train_corrupt}.json","w") as f: json.dump(out,f,indent=2)
    print(f"saved -> {OUT}/learned_inverter_{slug}_{args.train_corrupt}.json ({time.time()-t0:.0f}s)")

if __name__=="__main__": main()

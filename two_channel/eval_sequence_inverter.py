import argparse,json,os,time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from .sequence_inverter import SeqInv,beam_search,mech_log_likelihood
from .quotient_release import QuotientRelease
from .adjacency_builder_v2 import get_layer_block

def make_test(tok,n,sl,seed=43):
    from datasets import load_dataset
    ds=load_dataset("wikitext","wikitext-103-raw-v1",split="validation")
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--target_model",required=True)
    ap.add_argument("--target_layer",type=int,required=True)
    ap.add_argument("--quotient_ckpt",required=True)
    ap.add_argument("--inverter_ckpt",required=True)
    ap.add_argument("--r",type=int,required=True)
    ap.add_argument("--sigma_rel",type=float,default=0.2)
    ap.add_argument("--mech_id",type=int,default=0)
    ap.add_argument("--n_test",type=int,default=1000)
    ap.add_argument("--prefix_lens",type=str,default="8,16,32,64")
    ap.add_argument("--max_T",type=int,default=64)
    ap.add_argument("--dm",type=int,default=512)
    ap.add_argument("--beam",type=int,default=8)
    ap.add_argument("--lambda_rerank",type=float,default=1.0)
    ap.add_argument("--target_dtype",default="bfloat16")
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tdtype=getattr(torch,a.target_dtype)
    tok=AutoTokenizer.from_pretrained(a.target_model)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    target=AutoModelForCausalLM.from_pretrained(a.target_model,torch_dtype=tdtype).to(dev).eval()
    blk=get_layer_block(target,a.target_layer)
    d=target.config.hidden_size
    qr=QuotientRelease(d,a.r).to(dev).to(torch.float32).eval()
    qr.load_state_dict(torch.load(a.quotient_ckpt,map_location=dev))
    inv=SeqInv(r=a.r,vocab=tok.vocab_size,dm=a.dm,max_T=a.max_T).to(dev).eval()
    inv.load_state_dict(torch.load(a.inverter_ckpt,map_location=dev))
    BOS=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    results={}
    for sl in [int(x) for x in a.prefix_lens.split(",")]:
        ds=make_test(tok,a.n_test,sl)
        em=0;tok_correct=0;tok_total=0
        torch.manual_seed(0)
        bsz=8
        for i in range(0,len(ds),bsz):
            batch=ds[i:i+bsz]
            ids=torch.stack(batch).to(dev)
            cap=[None]
            def hk(m,i_,o,c=cap):
                c[0]=(o[0] if isinstance(o,tuple) else o).detach()
            h=blk.register_forward_hook(hk)
            with torch.no_grad():target(ids)
            h.remove()
            H=cap[0].float()
            with torch.no_grad():
                mu,ls=qr.enc(H)
                z=mu+torch.randn_like(mu)*(0.5*ls).exp()+a.sigma_rel*torch.randn_like(mu)
            mech=torch.full((z.size(0),),a.mech_id,device=dev,dtype=torch.long)
            sig=torch.zeros(z.size(0),device=dev,dtype=torch.long)
            seqs,scores=beam_search(inv,z,mech,sig,bos=BOS,eos=tok.eos_token_id,max_len=sl+1,B=a.beam)
            for j in range(len(batch)):
                cands=seqs[j]
                cand_ids=cands[:,1:sl+1]
                if a.lambda_rerank>0:
                    rerank=[]
                    for k in range(cand_ids.size(0)):
                        with torch.no_grad():
                            ll=mech_log_likelihood(qr,target,a.target_layer,cand_ids[k:k+1],z[j:j+1],a.sigma_rel)
                        rerank.append(ll.item())
                    rerank=torch.tensor(rerank,device=dev)
                    final_score=scores[j]+a.lambda_rerank*rerank
                    best=cand_ids[final_score.argmax()]
                else:
                    best=cand_ids[0]
                truth=batch[j]
                if torch.equal(best.cpu(),truth):em+=1
                tok_correct+=int((best.cpu()==truth).sum())
                tok_total+=len(truth)
        results[sl]=dict(exact_match=em/len(ds),token_acc=tok_correct/tok_total,n=len(ds))
        print(f"prefix_len={sl}: EM={em/len(ds):.4f} TA={tok_correct/tok_total:.4f}")
    out=dict(target_model=a.target_model,target_layer=a.target_layer,r=a.r,sigma_rel=a.sigma_rel,beam=a.beam,lambda_rerank=a.lambda_rerank,results=results)
    os.makedirs(os.path.dirname(a.out)or".",exist_ok=True)
    with open(a.out,"w") as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))

if __name__=="__main__":
    main()

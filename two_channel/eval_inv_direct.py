import argparse,json,os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer,AutoModelForCausalLM
from .sequence_inverter import SeqInv,beam_search
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
    ap.add_argument("--inverter_ckpt",required=True)
    ap.add_argument("--defense",choices=["clean","sigma_diag","isotropic"],default="sigma_diag")
    ap.add_argument("--sigma",type=float,default=5.0)
    ap.add_argument("--F_diag_path",default="")
    ap.add_argument("--n_test",type=int,default=500)
    ap.add_argument("--seq_len",type=int,default=32)
    ap.add_argument("--max_T",type=int,default=64)
    ap.add_argument("--dm",type=int,default=512)
    ap.add_argument("--beam",type=int,default=1)
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
    inv=SeqInv(r=d,vocab=tok.vocab_size,dm=a.dm,max_T=a.max_T).to(dev).eval()
    inv.load_state_dict(torch.load(a.inverter_ckpt,map_location=dev))
    F_diag=None
    if a.defense=="sigma_diag":
        F_diag=torch.load(a.F_diag_path,map_location=dev).float().clamp(min=1e-6) if a.F_diag_path and os.path.exists(a.F_diag_path) else torch.ones(d,device=dev)
    BOS=tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    print(f"defense={a.defense} sigma={a.sigma}")
    ds=make_test(tok,a.n_test,a.seq_len)
    em=0;tok_correct=0;tok_total=0
    bsz=8
    torch.manual_seed(0)
    for i in range(0,len(ds),bsz):
        batch=ds[i:i+bsz]
        ids=torch.stack(batch).to(dev)
        cap=[None]
        def hk(m,i_,o,c=cap):c[0]=(o[0] if isinstance(o,tuple) else o).detach()
        h=blk.register_forward_hook(hk)
        with torch.no_grad():target(ids)
        h.remove()
        H=cap[0].float()
        if a.defense=="isotropic":
            z=H+a.sigma*torch.randn_like(H)
        elif a.defense=="sigma_diag":
            std=a.sigma*F_diag.pow(-0.5)
            z=H+torch.randn_like(H)*std[None,None,:]
        else:
            z=H
        mech=torch.zeros(z.size(0),device=dev,dtype=torch.long)
        sig=torch.zeros(z.size(0),device=dev,dtype=torch.long)
        if a.beam>1:
            seqs,_=beam_search(inv,z,mech,sig,bos=BOS,eos=tok.eos_token_id,max_len=a.seq_len+1,B=a.beam)
            preds=seqs[:,0,1:a.seq_len+1]
        else:
            with torch.no_grad():
                cur=torch.full((z.size(0),1),BOS,device=dev,dtype=torch.long)
                for _ in range(a.seq_len):
                    logits=inv(z,mech,sig,cur)
                    nxt=logits[:,-1,:].argmax(-1,keepdim=True)
                    cur=torch.cat([cur,nxt],dim=1)
                preds=cur[:,1:a.seq_len+1]
        for j in range(len(batch)):
            best=preds[j].cpu()
            truth=batch[j]
            if torch.equal(best,truth):em+=1
            tok_correct+=int((best==truth).sum())
            tok_total+=len(truth)
    res=dict(target_model=a.target_model,target_layer=a.target_layer,defense=a.defense,sigma=a.sigma,seq_len=a.seq_len,n_test=len(ds),beam=a.beam,exact_match=em/len(ds),token_acc=tok_correct/tok_total)
    os.makedirs(os.path.dirname(a.out)or".",exist_ok=True)
    with open(a.out,"w") as f:json.dump(res,f,indent=2)
    print(json.dumps(res,indent=2))

if __name__=="__main__":
    main()

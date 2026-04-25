import torch
import torch.nn.functional as F
from collections import Counter

def get_layer_block(model,layer):
    if hasattr(model,'transformer') and hasattr(model.transformer,'h'):
        return model.transformer.h[layer]
    if hasattr(model,'model') and hasattr(model.model,'layers'):
        return model.model.layers[layer]
    raise ValueError("cannot find layers")

def hidden_at_layer(model,blk,ids):
    cap=[None]
    def hk(m,i,o,c=cap):
        c[0]=(o[0] if isinstance(o,tuple) else o).detach()
    h=blk.register_forward_hook(hk)
    with torch.no_grad():
        model(ids)
    h.remove()
    return cap[0][:,-1,:].float()

def hidden_with_logits(model,blk,ids):
    cap=[None]
    def hk(m,i,o,c=cap):
        c[0]=(o[0] if isinstance(o,tuple) else o).detach()
    h=blk.register_forward_hook(hk)
    with torch.no_grad():
        out=model(ids)
    h.remove()
    return cap[0][:,-1,:].float(),out.logits[:,-1,:].float()

def freq_bin(token_ids,n_bins=8):
    cnt=Counter(token_ids.tolist() if isinstance(token_ids,torch.Tensor) else token_ids)
    sorted_toks=sorted(cnt.items(),key=lambda x:-x[1])
    bins={}
    per=max(1,len(sorted_toks)//n_bins)
    for i,(t,_) in enumerate(sorted_toks):
        bins[t]=min(i//per,n_bins-1)
    return bins,n_bins

def build_random_subs(model,tok,blk,prefixes,n,vocab,device,seed=0):
    torch.manual_seed(seed)
    deltas=[]
    pf=prefixes[:max(64,n//8)]
    pos_per=max(1,n//len(pf))
    for p in pf:
        ids=p.unsqueeze(0).to(device)
        h_o=hidden_at_layer(model,blk,ids)
        T=ids.shape[1]
        for _ in range(pos_per):
            pos=torch.randint(1,T,(1,)).item()
            new=torch.randint(0,vocab,(1,)).item()
            mod=ids.clone()
            mod[0,pos]=new
            h_m=hidden_at_layer(model,blk,mod)
            deltas.append((h_o-h_m).cpu().squeeze(0))
            if len(deltas)>=n:break
        if len(deltas)>=n:break
    return torch.stack(deltas[:n])

def build_top_prob_subs(model,tok,blk,prefixes,n,k_alt,device,seed=1):
    torch.manual_seed(seed)
    deltas=[]
    pf=prefixes[:max(64,n//8)]
    pos_per=max(1,n//len(pf))
    for p in pf:
        ids=p.unsqueeze(0).to(device)
        with torch.no_grad():
            out=model(ids)
        logits_seq=out.logits[0]
        h_o=hidden_at_layer(model,blk,ids)
        T=ids.shape[1]
        for _ in range(pos_per):
            pos=torch.randint(1,T-1,(1,)).item()
            topk=torch.topk(logits_seq[pos-1],k_alt).indices.tolist()
            cur=ids[0,pos].item()
            cand=[t for t in topk if t!=cur]
            if not cand:continue
            new=cand[torch.randint(0,len(cand),(1,)).item()]
            mod=ids.clone()
            mod[0,pos]=new
            h_m=hidden_at_layer(model,blk,mod)
            deltas.append((h_o-h_m).cpu().squeeze(0))
            if len(deltas)>=n:break
        if len(deltas)>=n:break
    return torch.stack(deltas[:n])

def build_freqbin_subs(model,tok,blk,prefixes,n,device,seed=2,n_bins=8):
    torch.manual_seed(seed)
    all_toks=torch.cat([p for p in prefixes[:1000]])
    bins,n_bins=freq_bin(all_toks,n_bins)
    bins_inv={}
    for t,b in bins.items():bins_inv.setdefault(b,[]).append(t)
    deltas=[]
    pf=prefixes[:max(64,n//8)]
    pos_per=max(1,n//len(pf))
    for p in pf:
        ids=p.unsqueeze(0).to(device)
        h_o=hidden_at_layer(model,blk,ids)
        T=ids.shape[1]
        for _ in range(pos_per):
            pos=torch.randint(1,T,(1,)).item()
            cur=ids[0,pos].item()
            b=bins.get(cur,0)
            pool=bins_inv.get(b,[])
            cand=[t for t in pool if t!=cur]
            if not cand:continue
            new=cand[torch.randint(0,len(cand),(1,)).item()]
            mod=ids.clone()
            mod[0,pos]=new
            h_m=hidden_at_layer(model,blk,mod)
            deltas.append((h_o-h_m).cpu().squeeze(0))
            if len(deltas)>=n:break
        if len(deltas)>=n:break
    return torch.stack(deltas[:n])

def build_behavior_hard(model,tok,blk,prefixes,n,device,seed=3,n_pool=2000):
    torch.manual_seed(seed)
    pf=prefixes[:n_pool]
    H=[]
    P=[]
    for p in pf:
        ids=p.unsqueeze(0).to(device)
        h_o,lg=hidden_with_logits(model,blk,ids)
        H.append(h_o.cpu().squeeze(0))
        P.append(F.log_softmax(lg.cpu().squeeze(0),dim=-1))
    H=torch.stack(H)
    P=torch.stack(P)
    p_exp=P.exp()
    deltas=[]
    for i in range(len(pf)):
        if len(deltas)>=n:break
        idx=torch.randperm(len(pf))[:64]
        idx=idx[idx!=i]
        kls=(p_exp[i:i+1]*(P[i:i+1]-P[idx])).sum(-1)
        nn=idx[kls.argmin()].item()
        deltas.append(H[i]-H[nn])
    return torch.stack(deltas[:n])

def build_full_adjacency(model,tok,blk,prefixes,device,n_each=5000,k_alt=256):
    parts=[]
    parts.append(build_random_subs(model,tok,blk,prefixes,n_each,model.config.vocab_size,device))
    parts.append(build_top_prob_subs(model,tok,blk,prefixes,n_each,k_alt,device))
    parts.append(build_freqbin_subs(model,tok,blk,prefixes,n_each,device))
    parts.append(build_behavior_hard(model,tok,blk,prefixes,n_each,device))
    return torch.cat(parts,0),[p.shape[0] for p in parts]

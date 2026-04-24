"""Build the empirical adjacency set A for Renyi DP accounting.

For a Gaussian release of hidden state h, 'adjacent' pairs h, h' are pairs of
prefix states whose next-token distinguishability we care about. We expose
several strategies; the plan calls for

    top-512 model-prob + 512 random + 512 same-class + 512 behavior-hard + ground-truth

on 7B/14B, and full-vocab on GPT-2.
"""
import torch

def top_prob_neighbors(H,logits,k=512):
    n,d=H.shape
    p=logits.softmax(-1)
    out=[]
    for i in range(n):
        topk=torch.topk(p[i],k).indices
        out.append(H[i:i+1].repeat(len(topk),1)-H[topk])
    return torch.cat(out,0)

def random_neighbors(H,k=512,seed=0):
    n=H.shape[0]
    g=torch.Generator().manual_seed(seed)
    out=[]
    for i in range(n):
        j=torch.randint(0,n,(k,),generator=g)
        out.append(H[i:i+1].repeat(k,1)-H[j])
    return torch.cat(out,0)

def nearest_neighbors(H,k=512):
    n=H.shape[0]
    D=torch.cdist(H,H)
    out=[]
    for i in range(n):
        D[i,i]=float("inf")
        nn=D[i].topk(k,largest=False).indices
        out.append(H[i:i+1].repeat(k,1)-H[nn])
    return torch.cat(out,0)

def build_adjacency(H_bank,H_query,logits_query=None,n_per_query=None):
    pieces=[]
    if logits_query is not None:
        pieces.append(top_prob_neighbors(H_query,logits_query,k=min(512,logits_query.shape[1])))
    pieces.append(random_neighbors(H_query,k=512))
    pieces.append(nearest_neighbors(H_bank,k=min(512,H_bank.shape[0]-1)))
    Deltas=torch.cat(pieces,0)
    if n_per_query is not None:
        idx=torch.randperm(Deltas.shape[0])[:n_per_query]
        Deltas=Deltas[idx]
    return Deltas

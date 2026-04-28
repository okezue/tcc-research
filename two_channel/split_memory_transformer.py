import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSA(nn.Module):
    def __init__(self,d,h,drop=0.0):
        super().__init__()
        assert d%h==0
        self.h=h
        self.dh=d//h
        self.qkv=nn.Linear(d,3*d,bias=False)
        self.o=nn.Linear(d,d,bias=False)
        self.drop=drop
    def forward(self,x):
        import math
        B,T,d=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(B,T,self.h,self.dh).transpose(1,2)
        k=k.view(B,T,self.h,self.dh).transpose(1,2)
        v=v.view(B,T,self.h,self.dh).transpose(1,2)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dh)
        mask=torch.triu(torch.ones(T,T,device=scores.device,dtype=torch.bool),1)
        scores=scores.masked_fill(mask,float('-inf'))
        attn=F.softmax(scores,dim=-1)
        if self.drop>0 and self.training:
            attn=F.dropout(attn,p=self.drop)
        y=attn@v
        y=y.transpose(1,2).contiguous().view(B,T,d)
        return self.o(y)

class Block(nn.Module):
    def __init__(self,d,h,ff,drop=0.0):
        super().__init__()
        self.ln1=nn.LayerNorm(d)
        self.sa=CausalSA(d,h,drop)
        self.ln2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d))
    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x

class SMTLayer(nn.Module):
    def __init__(self,r,m,hr,hm,ff_r,ff_m,gamma_init=0.01):
        super().__init__()
        self.W_vu=nn.Linear(m,r,bias=False)
        self.W_uv=nn.Linear(r,m,bias=False)
        self.gamma=nn.Parameter(torch.tensor(gamma_init))
        self.lnu=nn.LayerNorm(r)
        self.lnv=nn.LayerNorm(m)
        self.bu=Block(r,hr,ff_r)
        self.bv=Block(m,hm,ff_m)
    def forward(self,u,v):
        u=u+self.bu(self.lnu(u+self.gamma*self.W_vu(v)))
        v=v+self.bv(self.lnv(v+self.W_uv(u)))
        return u,v

class SMT(nn.Module):
    def __init__(self,vocab,r=128,m=640,n_layers=12,hr=4,hm=4,ff_r=512,ff_m=1280,max_T=256):
        super().__init__()
        d=r+m
        self.r=r
        self.m=m
        self.tok=nn.Embedding(vocab,d)
        self.pos=nn.Embedding(max_T,d)
        self.layers=nn.ModuleList([SMTLayer(r,m,hr,hm,ff_r,ff_m) for _ in range(n_layers)])
        self.lnf=nn.LayerNorm(r)
        self.head=nn.Linear(r,vocab,bias=False)
        self.head.weight=nn.Parameter(self.tok.weight[:,:r].clone())
    def forward(self,ids,return_uv=False):
        B,T=ids.shape
        x=self.tok(ids)+self.pos(torch.arange(T,device=ids.device))[None]
        u,v=x[...,:self.r],x[...,self.r:]
        u_layers=[]
        v_layers=[]
        for L in self.layers:
            u,v=L(u,v)
            u_layers.append(u)
            v_layers.append(v)
        logits=self.head(self.lnf(u))
        if return_uv:
            return logits,u_layers,v_layers
        return logits

def hutchinson_logit_v_jac(model,ids,probe_layers):
    logits,u_l,v_l=model(ids,return_uv=True)
    B,T,V=logits.shape
    xi=torch.randn(B,T,V,device=logits.device)
    s=(logits*xi).sum()
    g=torch.autograd.grad(s,[v_l[li] for li in probe_layers],retain_graph=True,create_graph=True,allow_unused=True)
    pen=logits.new_zeros(())
    for gv in g:
        if gv is not None:
            pen=pen+gv.pow(2).sum()
    return pen

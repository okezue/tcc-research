"""Learned inversion model for hidden-state release.

Input : released activation h_tilde in R^d (or full sequence H_tilde in R^{T x d}).
Output: token sequence x_1, ..., x_T.

Architecture (following the plan):
  - ActivationEncoder: linear projection h_tilde -> Z in R^{M x D_enc}, M=16.
  - InverterDecoder: 6-layer transformer decoder, width=D_dec=512, 8 heads,
    causal self-attention + cross-attention to Z, tied token embeddings.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActivationEncoder(nn.Module):
    def __init__(self,d_in,d_enc=512,M=16):
        super().__init__()
        self.M=M;self.d_enc=d_enc
        self.proj=nn.Linear(d_in,M*d_enc)
        self.ln=nn.LayerNorm(d_enc)
    def forward(self,h):
        z=self.proj(h).reshape(h.shape[0],self.M,self.d_enc)
        return self.ln(z)

class DecBlock(nn.Module):
    def __init__(self,d,heads=8,ff=4):
        super().__init__()
        self.sa=nn.MultiheadAttention(d,heads,batch_first=True)
        self.ca=nn.MultiheadAttention(d,heads,batch_first=True)
        self.ln1=nn.LayerNorm(d);self.ln2=nn.LayerNorm(d);self.ln3=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,ff*d),nn.GELU(),nn.Linear(ff*d,d))
    def forward(self,x,z,mask=None):
        h=self.ln1(x)
        a,_=self.sa(h,h,h,attn_mask=mask,need_weights=False)
        x=x+a
        h=self.ln2(x)
        a,_=self.ca(h,z,z,need_weights=False)
        x=x+a
        x=x+self.ff(self.ln3(x))
        return x

class InverterDecoder(nn.Module):
    def __init__(self,d_in,vocab,d=512,heads=8,layers=6,M=16,T_max=64):
        super().__init__()
        self.enc=ActivationEncoder(d_in,d_enc=d,M=M)
        self.tok=nn.Embedding(vocab,d)
        self.pos=nn.Embedding(T_max,d)
        self.blocks=nn.ModuleList([DecBlock(d,heads=heads) for _ in range(layers)])
        self.ln=nn.LayerNorm(d)
        self.head_bias=nn.Parameter(torch.zeros(vocab))
        self.T_max=T_max
    def forward(self,h,x_prev):
        B,T=x_prev.shape
        z=self.enc(h)
        pos=torch.arange(T,device=x_prev.device)
        x=self.tok(x_prev)+self.pos(pos)[None]
        mask=torch.triu(torch.full((T,T),float("-inf"),device=x.device),diagonal=1)
        for b in self.blocks: x=b(x,z,mask=mask)
        x=self.ln(x)
        logits=x@self.tok.weight.T+self.head_bias
        return logits
    def loss(self,h,x):
        logits=self.forward(h,x[:,:-1])
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1))
    @torch.no_grad()
    def greedy(self,h,bos,T):
        B=h.shape[0]
        x=torch.full((B,1),bos,dtype=torch.long,device=h.device)
        for _ in range(T-1):
            lg=self.forward(h,x)
            nx=lg[:,-1].argmax(-1,keepdim=True)
            x=torch.cat([x,nx],1)
        return x

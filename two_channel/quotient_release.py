import torch
import torch.nn as nn
import torch.nn.functional as F
from .gradient_reversal import grad_reverse

class QEnc(nn.Module):
    def __init__(self,d,r):
        super().__init__()
        self.ln=nn.LayerNorm(d)
        self.f1=nn.Linear(d,4*r)
        self.f2=nn.Linear(4*r,2*r)
        self.r=r
    def forward(self,h):
        x=self.f2(F.gelu(self.f1(self.ln(h))))
        mu,ls=x.chunk(2,dim=-1)
        return mu,ls

class QDec(nn.Module):
    def __init__(self,d,r):
        super().__init__()
        self.f1=nn.Linear(r,4*r)
        self.f2=nn.Linear(4*r,d)
    def forward(self,z):
        return self.f2(F.gelu(self.f1(z)))

class RetHead(nn.Module):
    def __init__(self,r,d,proj=128):
        super().__init__()
        self.uz=nn.Linear(r,proj)
        self.vh=nn.Linear(d,proj)
    def forward(self,z_seq,h_seq):
        u=self.uz(z_seq.mean(dim=1))
        v=self.vh(h_seq.mean(dim=1))
        u=F.normalize(u,dim=-1)
        v=F.normalize(v,dim=-1)
        return u,v

def reparam(mu,ls):
    return mu+torch.randn_like(mu)*(0.5*ls).exp()

def kl_iso(mu,ls):
    return 0.5*(mu.pow(2)+ls.exp()-1.0-ls).sum(dim=-1).mean()

def info_nce(u,v,tau=0.1):
    s=(u@v.T)/tau
    lab=torch.arange(s.size(0),device=s.device)
    return F.cross_entropy(s,lab)

class QuotientRelease(nn.Module):
    def __init__(self,d,r):
        super().__init__()
        self.enc=QEnc(d,r)
        self.dec=QDec(d,r)
        self.ret=RetHead(r,d)
        self.r=r
    def forward(self,h_seq,sigma_rel=0.0,grl=1.0):
        B,T,d=h_seq.shape
        mu,ls=self.enc(h_seq)
        z=reparam(mu,ls)
        if sigma_rel>0:
            z=z+sigma_rel*torch.randn_like(z)
        h_hat=self.dec(z)
        z_adv=grad_reverse(z,grl)
        u,v=self.ret(z_adv,h_seq)
        return dict(z=z,h_hat=h_hat,mu=mu,ls=ls,u=u,v=v)

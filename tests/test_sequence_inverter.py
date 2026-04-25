import torch
from two_channel.sequence_inverter import SeqInv,beam_search

def test_forward():
    torch.manual_seed(0)
    V=100
    m=SeqInv(r=8,vocab=V,dm=64,nhead=4,enc_layers=2,dec_layers=2,ff=128,max_T=16,n_mech=4,n_sig=4)
    z=torch.randn(2,8,8)
    mech=torch.tensor([0,1])
    sig=torch.tensor([0,2])
    tgt=torch.randint(0,V,(2,5))
    o=m(z,mech,sig,tgt)
    assert o.shape==(2,5,V)
    assert torch.isfinite(o).all()

def test_beam():
    torch.manual_seed(0)
    V=20
    m=SeqInv(r=4,vocab=V,dm=32,nhead=4,enc_layers=1,dec_layers=1,ff=64,max_T=8,n_mech=2,n_sig=2)
    z=torch.randn(1,4,4)
    seqs,_=beam_search(m,z,torch.tensor([0]),torch.tensor([0]),bos=1,eos=2,max_len=6,B=4)
    assert seqs.shape[0]==1
    assert seqs.shape[1]==4
    assert seqs.shape[2]==6

def test_grad_flow():
    torch.manual_seed(0)
    V=40
    m=SeqInv(r=4,vocab=V,dm=32,nhead=2,enc_layers=1,dec_layers=1,ff=64,max_T=8,n_mech=2,n_sig=2)
    z=torch.randn(2,4,4)
    tgt=torch.randint(0,V,(2,3))
    out=m(z,torch.tensor([0,1]),torch.tensor([0,0]),tgt)
    L=out.sum()
    L.backward()
    has_grad=any(p.grad is not None and p.grad.abs().sum()>0 for p in m.parameters())
    assert has_grad

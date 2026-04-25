import torch
from two_channel.quotient_release import QuotientRelease,reparam,kl_iso,info_nce
from two_channel.gradient_reversal import grad_reverse

def test_forward_shapes():
    torch.manual_seed(0)
    B,T,d,r=4,8,128,16
    h=torch.randn(B,T,d)
    m=QuotientRelease(d,r)
    out=m(h,sigma_rel=0.1)
    assert out["z"].shape==(B,T,r)
    assert out["h_hat"].shape==(B,T,d)
    assert out["mu"].shape==(B,T,r)
    assert out["u"].shape==(B,128)
    assert out["v"].shape==(B,128)

def test_no_nan_at_init():
    torch.manual_seed(0)
    h=torch.randn(2,4,64)
    m=QuotientRelease(64,8)
    out=m(h,sigma_rel=0.1)
    for k,v in out.items():
        assert torch.isfinite(v).all(),k

def test_grad_reversal():
    x=torch.randn(3,4,requires_grad=True)
    y=grad_reverse(x,2.0)
    y.sum().backward()
    assert torch.allclose(x.grad,-2.0*torch.ones_like(x))

def test_kl_iso_zero_at_standard():
    mu=torch.zeros(2,4)
    ls=torch.zeros(2,4)
    assert kl_iso(mu,ls).abs()<1e-6

def test_info_nce_decreases_with_alignment():
    torch.manual_seed(0)
    u=torch.randn(8,16)
    v=torch.randn(8,16)
    L_rand=info_nce(u/u.norm(dim=-1,keepdim=True),v/v.norm(dim=-1,keepdim=True))
    u2=v.clone()
    L_aligned=info_nce(u2/u2.norm(dim=-1,keepdim=True),v/v.norm(dim=-1,keepdim=True))
    assert L_aligned<L_rand

def test_training_step_reduces_loss():
    torch.manual_seed(0)
    B,T,d,r=4,8,64,8
    h=torch.randn(B,T,d)
    m=QuotientRelease(d,r)
    opt=torch.optim.Adam(m.parameters(),lr=1e-2)
    losses=[]
    for _ in range(20):
        opt.zero_grad()
        out=m(h,sigma_rel=0.0)
        L=(out["h_hat"]-h).pow(2).mean()+1e-3*kl_iso(out["mu"],out["ls"])
        L.backward()
        opt.step()
        losses.append(L.item())
    assert losses[-1]<losses[0]

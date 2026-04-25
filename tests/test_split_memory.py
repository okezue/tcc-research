import torch
from two_channel.split_memory_transformer import SMT,hutchinson_logit_v_jac

def test_smt_forward():
    torch.manual_seed(0)
    V=64
    m=SMT(vocab=V,r=16,m=48,n_layers=2,hr=2,hm=2,ff_r=64,ff_m=96,max_T=16)
    ids=torch.randint(0,V,(2,8))
    o=m(ids)
    assert o.shape==(2,8,V)
    assert torch.isfinite(o).all()

def test_smt_uv_split():
    torch.manual_seed(0)
    V=32
    m=SMT(vocab=V,r=8,m=24,n_layers=2,hr=2,hm=2,ff_r=32,ff_m=48,max_T=8)
    ids=torch.randint(0,V,(1,4))
    logits,u_l,v_l=m(ids,return_uv=True)
    assert u_l[0].shape[-1]==8
    assert v_l[0].shape[-1]==24
    assert len(u_l)==2

def test_hutchinson_jacobian():
    torch.manual_seed(0)
    V=32
    m=SMT(vocab=V,r=8,m=24,n_layers=2,hr=2,hm=2,ff_r=32,ff_m=48,max_T=8)
    ids=torch.randint(0,V,(2,4))
    pen=hutchinson_logit_v_jac(m,ids,probe_layers=[0])
    assert torch.is_tensor(pen)
    assert torch.isfinite(pen).all()
    assert pen.requires_grad
    assert pen.item()>0

def test_smt_train_step():
    torch.manual_seed(0)
    V=32
    m=SMT(vocab=V,r=8,m=24,n_layers=2,hr=2,hm=2,ff_r=32,ff_m=48,max_T=8)
    opt=torch.optim.Adam(m.parameters(),lr=3e-4)
    ids=torch.randint(0,V,(4,8))
    losses=[]
    for _ in range(10):
        opt.zero_grad()
        logits=m(ids)
        L=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,V),ids[:,1:].reshape(-1))
        L.backward()
        opt.step()
        losses.append(L.item())
    assert losses[-1]<losses[0]

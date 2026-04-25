import torch
from two_channel.sigma_diag_validate import build_diag_sigma,worst_mahal

def test_alpha_one_minimax():
    torch.manual_seed(0)
    d=64
    F=torch.rand(d).mul(10).add(0.1)
    deltas=torch.randn(200,d)
    deltas=deltas/F.sqrt()
    alphas=[0.0,0.5,1.0,1.5]
    worsts=[worst_mahal(deltas,build_diag_sigma(F,a,1.0))[0] for a in alphas]
    best=alphas[worsts.index(min(worsts))]
    assert best==1.0

def test_kappa_invariance_of_alpha_optimum():
    torch.manual_seed(0)
    d=32
    F=torch.rand(d).mul(5).add(0.5)
    deltas=torch.randn(100,d)/F.sqrt()
    for kp in [0.5,1.0,2.0,5.0]:
        worsts=[]
        for a in [0.0,0.5,1.0,1.5]:
            s=build_diag_sigma(F,a,kp)
            worsts.append(worst_mahal(deltas,s)[0])
        assert worsts.index(min(worsts))==2

def test_utility_budget_respected():
    F=torch.rand(20).add(0.1)
    s=build_diag_sigma(F,1.0,3.0)
    budget=(F*s).sum().item()
    assert abs(budget-2*3.0)<1e-4

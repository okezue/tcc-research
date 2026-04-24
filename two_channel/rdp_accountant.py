"""Renyi DP accountant for Gaussian activation release.

Given a Gaussian mechanism M(h) = h + xi with xi ~ N(0,Sigma), the Renyi
divergence between releases at adjacent states h, h' with Delta = h - h' is

    D_alpha(M(h) || M(h')) = (alpha/2) * Delta^T Sigma^{-1} Delta.

Taking a worst-case adjacency set A we get the empirical RDP budget

    eps_alpha(Sigma) = (alpha/2) * max_{Delta in A} Delta^T Sigma^{-1} Delta,

which converts to (eps, delta)-DP via

    eps(delta) = min_{alpha>1} [eps_alpha + log(1/delta)/(alpha-1)].
"""
import torch,math

ALPHAS=(2,4,8,16,32,64,128)

def _sigma_inv(Sigma,eta_ratio=0.0,eps=1e-10):
    d=Sigma.shape[0]
    M=(Sigma+Sigma.T)/2
    if eta_ratio>0:
        eta=eta_ratio*Sigma.trace().item()/d
        M=M+eta*torch.eye(d,dtype=M.dtype,device=M.device)
    ev,U=torch.linalg.eigh(M.double())
    ev=ev.clamp(min=eps)
    return (U@torch.diag(1.0/ev)@U.T).float()

def max_mahalanobis(Deltas,Sigma,eta_ratio=0.0):
    Sinv=_sigma_inv(Sigma,eta_ratio=eta_ratio)
    vals=torch.einsum("nd,de,ne->n",Deltas,Sinv,Deltas)
    return float(vals.max())

def rdp_budget(Deltas,Sigma,alphas=ALPHAS,eta_ratio=0.0):
    s=max_mahalanobis(Deltas,Sigma,eta_ratio=eta_ratio)
    return {a:(a/2)*s for a in alphas}

def eps_delta(Deltas,Sigma,delta=1e-6,alphas=ALPHAS,eta_ratio=0.0):
    rdp=rdp_budget(Deltas,Sigma,alphas=alphas,eta_ratio=eta_ratio)
    best=math.inf;best_a=None
    for a,ea in rdp.items():
        v=ea+math.log(1.0/delta)/(a-1)
        if v<best: best=v;best_a=a
    return {"eps":best,"alpha_star":best_a,"rdp":rdp,"s_sigma":max_mahalanobis(Deltas,Sigma,eta_ratio=eta_ratio)}

def calibrate_scalar_to_eps(Deltas,Sigma0,eps_target,delta=1e-6,alphas=ALPHAS,eta_ratio=0.0):
    s0=max_mahalanobis(Deltas,Sigma0,eta_ratio=eta_ratio)
    best_c=None
    for a in alphas:
        rem=eps_target-math.log(1.0/delta)/(a-1)
        if rem<=0: continue
        c=(a/2)*s0/rem
        if c>0 and (best_c is None or c<best_c): best_c=c
    return best_c

"""Worst-case DP covariance SDP in reduced basis.

Plan spec: min_{Sigma >= 0, t} t  s.t.  Delta_i^T Sigma^{-1} Delta_i <= t  for all i,
                                       tr(F Sigma) <= kappa.

Schur: [Sigma  Delta_i; Delta_i^T  t] >= 0.

For large d we solve in a reduced basis U = [top-r eigvecs of F, of S, GE directions],
Sigma = U A U^T + eta * I.
"""
import torch

def sdp_worst_case(F,S,Deltas,kappa,r=128,eta=None):
    """Args: F (d,d) Fisher, S (d,d) margin cov, Deltas (n,d) adjacency set, kappa budget.
    Returns Sigma (d,d), t_star (float).
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise RuntimeError("cvxpy required: pip install cvxpy")
    import numpy as np
    d=F.shape[0]
    n=Deltas.shape[0]

    # Build reduced basis: top-r eigenvectors of F, top-r of S, top-r generalized
    F32=F.float();S32=S.float()
    evF,VF=torch.linalg.eigh(F32)
    idxF=evF.argsort(descending=True)[:r]
    VF=VF[:,idxF]
    evS,VS=torch.linalg.eigh(S32)
    idxS=evS.argsort(descending=True)[:r]
    VS=VS[:,idxS]
    # generalized: S v = lam F v => eigendecompose F^{-1/2} S F^{-1/2}
    F_lam=F32+(1e-3*F32.trace().item()/d)*torch.eye(d)
    evFL,UFL=torch.linalg.eigh(F_lam)
    F_isqrt=UFL@torch.diag(evFL.clamp(min=1e-8).rsqrt())@UFL.T
    M=F_isqrt@S32@F_isqrt
    evM,VM=torch.linalg.eigh((M+M.T)/2)
    idxM=evM.argsort(descending=True)[:r]
    VM=F_isqrt@VM[:,idxM]

    U_all=torch.cat([VF,VS,VM],dim=1)
    # orthonormalize
    Uq,_=torch.linalg.qr(U_all)
    U=Uq[:,:min(3*r,d)]
    r_eff=U.shape[1]

    if eta is None:
        eta=1e-3*F32.trace().item()/d

    # Project into reduced basis
    F_r=(U.T@F32@U).numpy()
    Deltas_r=(Deltas.float()@U).numpy()  # (n, r_eff)

    A=cp.Variable((r_eff,r_eff),symmetric=True)
    t=cp.Variable()
    constraints=[A>>0]
    # tr(F @ (U A U^T + eta*I)) <= kappa
    constraints+=[cp.trace(F_r@A)+eta*(F32.diag().sum().item())<=kappa]
    # Schur for each adjacency: [U A U^T  Delta; Delta^T  t] >= 0 in block form
    # reduced: Sigma_r = A + eta*(U^T U) = A + eta*I (since U orthonormal)
    import numpy as np
    I_r=np.eye(r_eff)
    Sigma_r=A+eta*I_r
    for i in range(n):
        delta=Deltas_r[i].reshape(-1,1)
        constraints+=[cp.bmat([[Sigma_r,delta],[delta.T,cp.reshape(t,(1,1))]])>>0]
    prob=cp.Problem(cp.Minimize(t),constraints)
    prob.solve(solver=cp.SCS)
    t_star=float(t.value)
    A_val=A.value
    Sigma_reduced=torch.tensor(A_val+eta*I_r,dtype=torch.float32)
    Sigma_full=U@Sigma_reduced@U.T+eta*torch.eye(d)
    return Sigma_full,t_star,r_eff

"""Mahalanobis-optimal defense covariance.

Solves

    min_{Sigma >= 0}  tr(S_rho @ Sigma^{-1})
    s.t.              tr(F_lam @ Sigma) <= kappa

where S_rho = S + rho*I, F_lam = F + lam*I, and the optimal Sigma is

    Sigma* = (kappa / tr(C^{1/2})) * F_lam^{-1/2} C^{1/2} F_lam^{-1/2}

with C = F_lam^{1/2} S_rho F_lam^{1/2}. The optimal objective is

    J*(kappa) = [tr(C^{1/2})]^2 / kappa.

See appendix_derivation for the proof.
"""
import torch


def _sym_sqrt(M,eps=1e-10):
    M=(M+M.T)/2
    evals,evecs=torch.linalg.eigh(M.double())
    evals=evals.clamp(min=eps)
    return (evecs@torch.diag(evals.sqrt())@evecs.T).float()


def _sym_inv_sqrt(M,eps=1e-10):
    M=(M+M.T)/2
    evals,evecs=torch.linalg.eigh(M.double())
    evals=evals.clamp(min=eps)
    return (evecs@torch.diag(evals.rsqrt())@evecs.T).float()


def solve_mahalanobis_optimal(F,S,kappa,lam_ratio=1e-3,rho_ratio=1e-3,eta_ratio=0.0):
    """Solve the Mahalanobis-optimal defense covariance.

    Arguments:
        F        -- hidden-state Fisher / gradient covariance (d x d)
        S        -- margin-direction covariance (d x d)
        kappa    -- utility budget tr(F_lam @ Sigma) <= kappa
        lam_ratio -- Fisher ridge as fraction of tr(F)/d (default 1e-3)
        rho_ratio -- S ridge as fraction of tr(S)/d (default 1e-3)
        eta_ratio -- isotropic floor on Sigma as fraction of tr(Sigma*)/d (default 0 = no floor)

    Returns: dict with Sigma_star, J_star, G_Mah scalar, trace diagnostics.
    """
    d=F.shape[0]
    F=F.float(); S=S.float()
    F_lam=F+(lam_ratio*F.trace().item()/d)*torch.eye(d)
    S_rho=S+(rho_ratio*S.trace().item()/d)*torch.eye(d)
    F_half=_sym_sqrt(F_lam)
    F_inv_half=_sym_inv_sqrt(F_lam)
    C=F_half@S_rho@F_half
    C=(C+C.T)/2
    C_half=_sym_sqrt(C)
    tr_C_half=float(C_half.trace())
    Sigma_star=(kappa/tr_C_half)*F_inv_half@C_half@F_inv_half
    Sigma_star=(Sigma_star+Sigma_star.T)/2
    J_star=(tr_C_half**2)/kappa
    tr_F=float(F_lam.trace()); tr_S=float(S_rho.trace())
    J_iso=tr_F*tr_S/kappa
    G_Mah=J_iso/J_star
    if eta_ratio>0:
        eta=eta_ratio*Sigma_star.trace().item()/d
        Sigma_star=Sigma_star+eta*torch.eye(d)
    return {
        "Sigma_star":Sigma_star,
        "J_star":J_star,
        "J_iso":J_iso,
        "G_Mah":G_Mah,
        "tr_C_half":tr_C_half,
        "tr_F_lam":tr_F,
        "tr_S_rho":tr_S,
    }


def gen_eigen_gain(F,S,k,lam_ratio=1e-3,rho_ratio=1e-3):
    """Compute the Euclidean-attacker gain G_Euc = lambda_top_k_avg / lambda_avg.

    lambda_avg = tr(S) / tr(F); lambda_top_k_avg is mean of top-k generalized eigenvalues
    of S_rho v = lambda F_lam v.
    """
    d=F.shape[0]
    F_lam=F+(lam_ratio*F.trace().item()/d)*torch.eye(d)
    S_rho=S+(rho_ratio*S.trace().item()/d)*torch.eye(d)
    F_inv_half=_sym_inv_sqrt(F_lam)
    M=F_inv_half@S_rho@F_inv_half
    M=(M+M.T)/2
    evals=torch.linalg.eigvalsh(M).sort(descending=True).values
    lam_top=float(evals[:k].mean())
    lam_avg=float(S_rho.trace()/F_lam.trace())
    return {"G_Euc":lam_top/lam_avg,"lam_top_k_avg":lam_top,"lam_avg":lam_avg,"top_eigvals":evals[:16].tolist()}


def sample_gaussian_with_cov(Sigma,seed=None,n_samples=1):
    """Sample z ~ N(0, Sigma). Uses eigendecomposition to handle rank-deficient Sigma."""
    d=Sigma.shape[0]
    if seed is not None:
        g=torch.Generator().manual_seed(seed)
        z=torch.randn(n_samples,d,generator=g)
    else:
        z=torch.randn(n_samples,d)
    evals,evecs=torch.linalg.eigh((Sigma+Sigma.T)/2)
    evals=evals.clamp(min=0)
    L=evecs@torch.diag(evals.sqrt())
    return z@L.T if n_samples>1 else (z@L.T).squeeze(0)

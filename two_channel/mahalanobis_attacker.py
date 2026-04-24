"""Defense-adaptive Mahalanobis retrieval attacker.

Given a noised query tilde_h and a clean bank, compute retrieval score

    d(tilde_h, c) = (tilde_h - c)^T (Sigma + tau*I)^{-1} (tilde_h - c)

where Sigma is the exact covariance the defender used and tau is a ridge
shrinkage floor. The shrinkage variant tunes tau on a validation split.
"""
import torch

_EIGH_CACHE={}

def _cached_eigh(Sigma,eps=1e-10):
    key=(id(Sigma),Sigma.shape)
    c=_EIGH_CACHE.get(key)
    if c is not None: return c
    M=(Sigma+Sigma.T)/2
    ev,U=torch.linalg.eigh(M.float())
    _EIGH_CACHE[key]=(ev,U)
    return ev,U

def _whiten_matrix(Sigma,tau,eps=1e-10):
    ev,U=_cached_eigh(Sigma)
    d=ev.shape[0]
    inv_sqrt=(ev+tau).clamp(min=eps).rsqrt()
    return (U*inv_sqrt.unsqueeze(0))@U.T


def mahalanobis_retrieval(H_q,H_bank,Sigma,tau,query_idx):
    """Rank bank entries by (q-c)^T (Sigma + tau I)^{-1} (q-c).

    Arguments:
        H_q       -- (n_q, d) query states (noised)
        H_bank    -- (n_b, d) clean candidate bank
        Sigma     -- (d, d) defender covariance
        tau       -- scalar ridge
        query_idx -- (n_q,) indices into H_bank of each query's true match

    Returns: dict with top1, mrr, median_rank, ranks_array.
    """
    L=_whiten_matrix(Sigma,tau)
    Hq=H_q@L.T
    Hb=H_bank@L.T
    D=torch.cdist(Hq,Hb)
    n_q=H_q.shape[0]
    ranks=[]
    for i in range(n_q):
        qi=query_idx[i]
        row=D[i]
        rank=(row<row[qi]).sum().item()+1
        ranks.append(rank)
    return {
        "top1":sum(1 for r in ranks if r==1)/n_q,
        "mrr":sum(1.0/r for r in ranks)/n_q,
        "med_rank":float(sorted(ranks)[len(ranks)//2]),
        "ranks":ranks,
    }


def tune_tau(H_val,H_bank,Sigma,val_idx,tau_grid=None):
    """Tune the Mahalanobis shrinkage floor tau on a validation split."""
    if tau_grid is None:
        base=Sigma.trace().item()/Sigma.shape[0]
        tau_grid=[0.0,1e-3*base,1e-2*base,1e-1*base,base,1e1*base]
    best_tau=None;best_mrr=0
    for tau in tau_grid:
        r=mahalanobis_retrieval(H_val,H_bank,Sigma,tau,val_idx)
        if r["mrr"]>best_mrr:
            best_mrr=r["mrr"]
            best_tau=tau
    return best_tau,best_mrr


def l2_retrieval(H_q,H_bank,query_idx):
    """Plain full-space L2 retrieval (baseline)."""
    D=torch.cdist(H_q,H_bank)
    n_q=H_q.shape[0]
    ranks=[]
    for i in range(n_q):
        qi=query_idx[i]
        row=D[i]
        rank=(row<row[qi]).sum().item()+1
        ranks.append(rank)
    return {
        "top1":sum(1 for r in ranks if r==1)/n_q,
        "mrr":sum(1.0/r for r in ranks)/n_q,
        "med_rank":float(sorted(ranks)[len(ranks)//2]),
        "ranks":ranks,
    }


def subspace_retrieval(H_q,H_bank,P,query_idx):
    """Retrieval in the subspace spanned by projector P (used for P_I / P_B attackers)."""
    L=_whiten_matrix_from_projector(P)
    Hq=H_q@L.T
    Hb=H_bank@L.T
    D=torch.cdist(Hq,Hb)
    n_q=H_q.shape[0]
    ranks=[]
    for i in range(n_q):
        qi=query_idx[i]
        row=D[i]
        rank=(row<row[qi]).sum().item()+1
        ranks.append(rank)
    return {
        "top1":sum(1 for r in ranks if r==1)/n_q,
        "mrr":sum(1.0/r for r in ranks)/n_q,
        "med_rank":float(sorted(ranks)[len(ranks)//2]),
        "ranks":ranks,
    }


def _whiten_matrix_from_projector(P):
    """For a projector P, return L such that L^T L = P (whitening inside the subspace)."""
    evals,evecs=torch.linalg.eigh((P+P.T)/2)
    evals=evals.clamp(min=0)
    return (evecs@torch.diag(evals.sqrt())@evecs.T).float()

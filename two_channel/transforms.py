import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass

@dataclass
class QuantStats:
    maxabs: torch.Tensor

def compute_quant_stats(h_set: torch.Tensor) -> QuantStats:
    return QuantStats(maxabs=h_set.abs().amax(dim=0))

def quantize(x: torch.Tensor, bits: int, stats: QuantStats) -> torch.Tensor:
    if bits>=32:
        return x
    qmax=2**(bits-1)-1
    alpha=stats.maxabs.to(x.device)/(qmax+1e-12)
    q=torch.clamp(torch.round(x/alpha),-qmax,qmax)
    return q*alpha

def add_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma<=0:
        return x
    rms=x.norm()/max(x.numel()**0.5,1)
    return x+sigma*rms*torch.randn_like(x)

def project(h: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    return h@U@U.T

def project_complement(h: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    return h-project(h,U)

class Transform:
    def __init__(self, U: torch.Tensor, mode: str="behavior",
                 bits: int=32, sigma: float=0.0,
                 stats: Optional[QuantStats]=None):
        self.U=U
        self.mode=mode
        self.bits=bits
        self.sigma=sigma
        self.stats=stats

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode=="behavior":
            p=project(h,self.U)
        elif self.mode=="identity":
            p=project_complement(h,self.U)
        elif self.mode=="random":
            p=project(h,self.U)
        elif self.mode=="full":
            p=h
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.bits<32 and self.stats is not None:
            p=quantize(p,self.bits,self.stats)
        if self.sigma>0:
            p=add_noise(p,self.sigma)
        return p

    def quantize_to_codes(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode=="behavior":
            p=project(h,self.U)
        elif self.mode=="identity":
            p=project_complement(h,self.U)
        elif self.mode=="random":
            p=project(h,self.U)
        elif self.mode=="full":
            p=h
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        if self.bits>=32 or self.stats is None:
            return p
        qmax=2**(self.bits-1)-1
        alpha=self.stats.maxabs.to(p.device)/(qmax+1e-12)
        return torch.clamp(torch.round(p/alpha),-qmax,qmax).long()

class HookTransform(nn.Module):
    def __init__(self, transform_fn):
        super().__init__()
        self.transform_fn=transform_fn
        self.handle=None

    def register(self, block):
        def hook(module, input, output):
            if isinstance(output,tuple):
                h=output[0]
                h_t=self.transform_fn(h)
                return (h_t,)+output[1:]
            return self.transform_fn(output)
        self.handle=block.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle=None

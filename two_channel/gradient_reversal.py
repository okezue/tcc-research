import torch
class _GR(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,l):
        ctx.l=l
        return x.view_as(x)
    @staticmethod
    def backward(ctx,g):
        return -ctx.l*g,None
def grad_reverse(x,l=1.0):
    return _GR.apply(x,l)

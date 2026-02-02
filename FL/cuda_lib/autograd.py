import torch
from . import atten_cuda


class ForwardEmissionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atten, C, em_cs, theta):
        """
        atten: (n_angle, H, W)
        C:     (n_ref, H, W)
        em_cs: (n_angle, n_ref)
        theta: (n_angle)
        """
        ctx.save_for_backward(atten, em_cs, theta)
        Pf = atten_cuda.forward_emission(atten, C, em_cs, theta)
        return Pf

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: ∂L/∂Pf, shape (n_angle, W)
        """
        atten, em_cs, theta = ctx.saved_tensors

        # Hᵀ (∂L/∂Pf)
        grad_C = atten_cuda.backward_emission(
            atten,
            grad_output.contiguous(),
            em_cs,
            theta,
        )

        # Only C gets gradient
        return None, grad_C, None, None


def forward_emission_autograd(atten, C, em_cs, theta):
    return ForwardEmissionFunction.apply(atten, C, em_cs, theta)

class ForwardEmissionBatchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, atten, C, em_cs, theta):
        """
        atten: (n_angle, B, H, W)
        C:     (n_ref,   B, H, W)
        em_cs: (n_angle, n_ref)
        theta: (n_angle)
        """
        ctx.save_for_backward(atten, em_cs, theta)

        Pf = atten_cuda.forward_emission_batch(
            atten,
            C,
            em_cs,
            theta,
        )
        return Pf

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: ∂L/∂Pf, shape (n_angle, B, W)
        """
        atten, em_cs, theta = ctx.saved_tensors

        grad_C = atten_cuda.backward_emission_batch(
            atten,
            grad_output.contiguous(),
            em_cs,
            theta,
        )

        # Only C gets gradient
        return None, grad_C, None, None

def forward_emission_batch_autograd(atten, C, em_cs, theta):
    return ForwardEmissionBatchFunction.apply(atten, C, em_cs, theta)

import torch

def poisson_loss(Pf, I):
    return (Pf - I * torch.log(Pf + 1e-6)).mean()

def poisson_loss_image(pred, target, eps=1e-6):
    pred = pred.clamp_min(eps)
    return (pred - target * torch.log(pred)).mean()

def denoiser_consistency_loss(C, C_d):
    return ((C_d - C) ** 2).mean()

def gradient_loss(x):
    dx = x[..., 1:, :] - x[..., :-1, :]
    dy = x[..., :, 1:] - x[..., :, :-1]
    return (dx.abs().mean() + dy.abs().mean())

def jacobian_regularization(denoiser, C, eps=1e-3):
    noise = torch.randn_like(C)
    C_perturbed = C + eps * noise

    D1 = denoiser(C)
    D2 = denoiser(C_perturbed)

    num = (D2 - D1).norm()
    den = (eps * noise).norm() + 1e-6
    return num / den
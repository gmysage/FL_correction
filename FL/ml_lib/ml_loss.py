import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp
import math
import numpy as np
from torch.nn import MSELoss

def mse_loss(input, target):
    mseloss = MSELoss()
    return mseloss(input, target)


def l1_loss(input, target):
    loss_fn = nn.L1Loss()
    return loss_fn(input, target)

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

def tv_loss(x):
    return gradient_loss(x)

def tv_loss_norm(x):
    n = torch.numel(x)
    loss = tv_loss(x) / n
    return loss


def jacobian_regularization(denoiser, C, eps=1e-3):
    noise = torch.randn_like(C)
    C_perturbed = C + eps * noise

    D1 = denoiser(C)
    D2 = denoiser(C_perturbed)

    num = (D2 - D1).norm()
    den = (eps * noise).norm() + 1e-6
    return num / den

def l1_loss(inputs, targets):
    loss = nn.L1Loss()
    output = loss(inputs, targets)
    return output

def psnr(label, outputs, max_val=1):
    try:
        label = label.cpu().detach().numpy()
    except:
        pass
    try:
        outputs = outputs.cpu().detach().numpy()
    except:
        pass
    max_val = max(np.max(label), np.max(outputs))
    img_diff = (outputs - label) / max_val
    rmse = math.sqrt(np.mean((img_diff)**2))
    if rmse == 0:
        return 100
    else:
        PSNR = 20 * math.log10(1. / rmse)
        return PSNR

## ssim loss
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim_loss(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def check_dic_key(dic, key):
    if key in dic:
        return True
    else:
        return False

def extract_h_loss(h_loss, loss_summary, loss_r):
    txt = ''

    for k in h_loss.keys():
        if 'psnr' in k:
            continue
        value = loss_summary[k]
        prefix = 'Y' if loss_r[k] else 'N'
        rate = loss_r[k]

        h_loss[k]['value'].append(value)
        h_loss[k]['rate'].append(rate)
        txt = txt + f' ({prefix}) ({rate:.1e}) {k} = {value:.3e}\n'

    for k in loss_summary.keys():
        if 'psnr' in k:
            if not check_dic_key(h_loss, k):
                h_loss[k] = []
            h_loss[k].append(loss_summary[k])
            txt = txt + f'{k} = {loss_summary[k]:2.3f}\n'

    current_psnr = loss_summary["psnr"]
    return h_loss, txt, current_psnr

####
def get_features_vgg19(image, model_feature, layers=None):
    if layers is None:
        layers = {'2': 'conv1_2',
                  '7': 'conv2_2',
                  '16': 'conv3_4',
                  '25': 'conv4_4'
                 }
    features = {}
    x = image
    for idx, layer in enumerate(model_feature):
        x = layer(x)
        if str(idx) in layers:
            features[layers[str(idx)]] = x
    return features


def vgg_loss(outputs, label, vgg19, device='cuda'):
    # Ensure inputs are tensors and handle device placement cleanly
    out = outputs.to(device) if torch.is_tensor(outputs) else torch.tensor(outputs, dtype=torch.float32, device=device)
    lab = label.to(device).detach() if torch.is_tensor(label) else torch.tensor(label, dtype=torch.float32,
                                                                                device=device).detach()

    # Normalize by the max value of the ground truth safely
    lab_max = torch.max(lab) + 1e-8
    out = out / lab_max
    lab = lab / lab_max

    # Broadcast single channel grayscale to 3 channels for VGG19
    if out.shape[1] == 1:
        out = out.repeat(1, 3, 1, 1)
    if lab.shape[1] == 1:
        lab = lab.repeat(1, 3, 1, 1)

    # Extract VGG activations (Gradients will flow natively through out features)
    feature_out1 = 0.5 * get_features_vgg19(out, vgg19, {'2': 'conv1_2'})['conv1_2']
    feature_out2 = 0.5 * get_features_vgg19(out, vgg19, {'25': 'conv4_4'})['conv4_4']
    feature_lab1 = 0.5 * get_features_vgg19(lab, vgg19, {'2': 'conv1_2'})['conv1_2']
    feature_lab2 = 0.5 * get_features_vgg19(lab, vgg19, {'25': 'conv4_4'})['conv4_4']

    # Directly calculate and return the loss value tensor
    return nn.MSELoss()(feature_out1, feature_lab1) + nn.MSELoss()(feature_out2, feature_lab2)

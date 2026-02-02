import torch
import numpy as np
from .cuda_lib.atten_cuda import mlem_cuda, mlem_cuda_batch

def torch_mlem_recon(C_init,        # (n_ref, H, W)
                    prj_sli,        # (n_angle, W)
                    angle_list,     # (n_angle)
                    atten = None,   # (n_angle, H, W)
                    em_cs = None,   # (n_angle, n_ref)
                    rho = 1,
                    pix = 1,
                    n_iter = 50,
                    beta = 1e-3,
                    delta = 0.01,
                    device = 'cuda'
                    ):
    n_ref, H, W = C_init.shape
    n_angle = len(angle_list)
    theta = angle_list / 180. * np.pi
    theta_cuda = torch.tensor(theta, dtype=torch.float, device=device)
    if atten is None:
        atten_cuda = torch.ones(n_angle, H, W, dtype=torch.float, device=device)
    else:
        atten_cuda = torch.tensor(atten, dtype=torch.float, device=device)
    
    if em_cs is None:
        em_cs = np.ones((n_angle, n_ref))
    else:        
        if len(em_cs.shape) == 1:
            em_cs = em_cs[:, np.newaxis] 
    em_cs = em_cs * rho * pix
    em_cs_cuda = torch.tensor(em_cs, dtype=torch.float, device=device)
    
    C_cuda = torch.tensor(C_init, dtype=torch.float, device=device)
    I_cuda = torch.tensor(prj_sli, dtype=torch.float, device=device)

    rec = mlem_cuda(C_cuda,
                    atten_cuda,
                    em_cs_cuda,
                    theta_cuda,
                    I_cuda,
                    n_iter,
                    beta,
                    delta
                    )
    rec = rec.cpu().numpy()
    return rec


def torch_mlem_recon_batch(C_init,  # (n_ref, n_sli, H, W)
                    prjs,           # (n_angle, n_sli, W)
                    angle_list,     # (n_angle)
                    atten = None,   # (n_angle, n_sli, H, W)
                    em_cs = None,   # (n_angle, n_ref)
                    rho = 1,
                    pix = 1,
                    n_iter = 50,
                    beta = 1e-3,
                    delta = 0.01,
                    device = 'cuda'
                    ):
    if len(C_init.shape) == 3:
        C_init = C_init[:, np.newaxis]
    if len(prjs.shape) == 2:
        prjs = prjs[:, np.newaxis]
            
    n_ref, n_sli, H, W = C_init.shape
    n_angle = len(angle_list)
    theta = angle_list / 180. * np.pi
    theta_cuda = torch.tensor(theta, dtype=torch.float, device=device)
    if atten is None:
        atten_cuda = torch.ones(n_angle, n_sli, H, W, dtype=torch.float, device=device)
    else:
        if len(atten.shape) == 3:
            atten = atten[:, np.newaxis]
        atten_cuda = torch.tensor(atten, dtype=torch.float, device=device)
    
    if em_cs is None:
        em_cs = np.ones((n_angle, n_ref))
    else:        
        if len(em_cs.shape) == 1:
            em_cs = em_cs[:, np.newaxis] 
    em_cs = em_cs * rho * pix
    em_cs_cuda = torch.tensor(em_cs, dtype=torch.float, device=device)
    
    C_cuda = torch.tensor(C_init, dtype=torch.float, device=device)
    I_cuda = torch.tensor(prjs, dtype=torch.float, device=device)

    rec = mlem_cuda_batch(C_cuda,
                          atten_cuda,
                          em_cs_cuda,
                          theta_cuda,
                          I_cuda,
                          n_iter,
                          beta,
                          delta
                          )
    rec = rec.cpu().numpy()
    return rec
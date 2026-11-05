from FL_correction_core import mlem_matrix, generate_H, re_projection, row_sum
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

def demo_vinillar_mlem_recon():
    fn_root = '/media/mingyuan/Data/FL_correction/FL/legacy/legacy_example'
    Ni = io.imread(fn_root + '/gt_image/gt_Ni.tiff') # (50, 160, 160)
    angle_list = np.loadtxt(fn_root + '/angle_list.txt') # (180,)
    shape_2D = Ni[0].shape # (160, 160)
    prj3D = re_projection(Ni, angle_list) #(180, 50, 160)
    img2D = np.ones(shape_2D) # (160, 160)
    I_2D = prj3D[:, 25].flatten() # (28800,)
    H = generate_H('Ni', 25, angle_list, flag=0, shape_2D=shape_2D) # (28800, 25600)
    img_cor = mlem_matrix(img2D, H, I_2D, n_iter=20)
    plt.figure()
    plt.imshow(img_cor, clim=[0,0.8])

    """
    ####################
    # explicit mlem
    ####################
    
    p = H
    y = I_2D
    n_iter = 20
    
    A_new = img2D.flatten().astype(np.float32)    # convert 2D array in 1d array
    A_new = A_new.reshape((len(A_new),1))
    b_sum = row_sum(p)     #b_sum = np.sum(p, axis=0)
    y[y<0] = 0
    if len(y.shape) == 1:
        y = y.reshape((len(y), 1))
    for _ in trange(n_iter):
        Pf = p @ A_new
        Pf[Pf < 1e-6] = 1
        t1 = p
        t2 = y / Pf
        a_sum = t2.T @ t1    
        A_new = A_new * (a_sum / b_sum).T
        A_new[np.isnan(A_new)] = 0
    img_cor = np.reshape(A_new, img2D.shape)
    """

    """
    ####################
    # explicit mess-up mlem
    ####################

    p = H
    y = I_2D
    n_iter = 20

    A_new = img2D.flatten().astype(np.float32)    # convert 2D array in 1d array
    A_new = A_new.reshape((len(A_new),1))
    b_sum = row_sum(p)     #b_sum = np.sum(p, axis=0)
    y[y<0] = 0
    if len(y.shape) == 1:
        y = y.reshape((len(y), 1))
    for _ in trange(n_iter):
        Pf = p @ A_new
        Pf[Pf < 1e-6] = 1
        t1 = p
        t2 = y / Pf
        a_sum = t2.T @ t1    
        aa_sum = a_sum.reshape(shape_2D)
        aa_sum = aa_sum[::-1]
        a_sum = aa_sum.reshape(a_sum.shape)
        A_new = A_new * (a_sum / b_sum).T
        A_new[np.isnan(A_new)] = 0
    img_cor_mess = np.reshape(A_new, img2D.shape)
    plt.figure()
    plt.imshow(img_cor_mess)
    plt.colorbar()
    """
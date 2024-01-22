from tqdm import trange
from skimage import io
from .image_util import *
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

def mk_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_attenuation(angle_id, fpath_atten, elem):
    #fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_att = fpath_atten + f'/atten_{elem}_prj_{angle_id:04d}.tiff'
    coef_att = io.imread(fn_att)
    return coef_att


def read_projection(angle_id, fpath_atten, elem):
    #fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_prj = fpath_atten + f'/{elem}_ref_prj_{angle_id:04d}.tiff'
    img_prj = io.imread(fn_prj)
    return img_prj


def read_attenuation_at_all_angle(angle_list, fpath_atten, elem):
    print('reading attenuation files ...')
    n = len(angle_list)
    for i in trange(n):
        tmp = read_attenuation(i, fpath_atten, elem)
        if i == 0: 
            s = tmp.shape
            coef_att4D = np.zeros((n, *s))
        coef_att4D[i] = tmp
    return coef_att4D

def read_recon(fpath_recon, iter_id, elem):
    fn = fpath_recon + f'/{elem}_iter_{iter_id:02d}.tiff'
    tmp = io.imread(fn)
    return tmp

def read_recon_all_elem(fn_root, iter_id, elem_type):
    n = len(elem_type)
    for i, elem in enumerate(elem_type):
        #tmp = read_recon(fn_root, iter_id, elem)
        fpath_recon = fn_root + '/recon'
        tmp = read_recon(fpath_recon, iter_id, elem)
        if i == 0:
            s = tmp.shape
            recon4D = np.zeros([n, *s])
        recon4D[i] = tmp
    return recon4D

def save_recon(fn_root, recon_cor, elem_type, iter_id):
    from skimage import io
    fsave_root = fn_root + f'/recon'
    mk_directory(fsave_root)
    fn_cor_save = fn_root + f'/recon/recon_{iter_id:02d}.h5'
    with h5py.File(fn_cor_save, 'w') as hf:
        for i, key in enumerate(elem_type):
            hf.create_dataset(key, data=recon_cor[i].astype(np.float32))
    for i, elem in enumerate(elem_type):
        fsave_tiff = fn_root + f'/recon/{elem}_iter_{iter_id:02d}.tiff'
        io.imsave(fsave_tiff, recon_cor[i].astype(np.float32))


def inspect_recon(recon_raw, iter_id, fn_root, elem, angle_list, ang_idx=0, sli=None):
    fpath_atten = fn_root + f'/Angle_prj_{iter_id:02d}'
    ang = angle_list[ang_idx]
    n = len(angle_list)
    s_tomo = recon_raw.shape
    img_raw_rot = rot3D(recon_raw, ang)
    recon_correction = read_recon(fn_root+'/recon', iter_id, elem)
    img_rot = FL.rot3D(recon_correction, ang)
    if sli is None:
        sli = recon_correction.shape[0]//2

    coef_att = read_attenuation(ang_idx, fpath_atten, elem)
    img_prj = read_projection(ang_idx, fpath_atten, elem)

    img_att = (coef_att * img_rot)
    img_reprj = np.sum(img_att, axis=1)
    clim = [0, np.max(img_reprj)*1.01]

    plt.figure(figsize=(18, 8))
    plt.suptitle(f'{elem} (iter={iter_id}): comparison at slice={sli}')

    plt.subplot(2,3,1);    plt.imshow(img_raw_rot[sli], cmap='turbo');    plt.colorbar()
    plt.title(f'tomo: raw reconstruction, angle={ang}')

    plt.subplot(2,3,2);    plt.imshow(img_prj, clim=clim, cmap='twilight_r');    
    plt.colorbar();    plt.title(f'proj: measured (raw), angle={ang} ')

    plt.subplot(2,3,4);    plt.imshow(img_rot[sli], cmap='turbo');    plt.colorbar()
    plt.title(f'tomo: corrected, angle={ang}')
    
    plt.subplot(2,3,5);    plt.imshow(img_reprj, clim=clim, cmap='twilight_r');    
    plt.colorbar();    plt.title(f'proj: attenu-simulated, angle={ang}')
        
    # inspect sino
    sino_path = fpath_atten + f'/{elem}_ref_prj*'
    fn_sino_raw = np.sort(glob.glob(sino_path))
    sino_raw = np.zeros((n, s_tomo[2]))
    sino_cor = np.zeros(sino_raw.shape)
    for i in range(n):
        tmp = io.imread(fn_sino_raw[i])[sli]
        sino_raw[i] = tmp.copy()
        tmp_rot = rot3D(recon_correction[sli], angle_list[i])
        sino_cor[i] = np.sum(tmp_rot, axis=1)

    plt.subplot(2,3,3);    plt.imshow(sino_raw, cmap='viridis');    plt.colorbar()
    plt.title(f'sino: measured (raw)')
    
    plt.subplot(2,3,6);    plt.imshow(sino_cor, cmap='viridis')    
    plt.colorbar();    plt.title(f'sino: corrected')
    
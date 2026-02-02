from skimage import io
from tqdm import tqdm, trange
from .util import mk_directory, save_dict_to_hdf5
from .image_util import rot3D
import matplotlib.pyplot as plt
import numpy as np
import glob
import h5py

############################
#### read attenunation #####
############################
def read_attenuation(angle_id, fpath_atten, elem):
    # fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_att = fpath_atten + f'/atten_{elem}_prj_{angle_id:04d}.tiff'
    coef_att = io.imread(fn_att)
    return coef_att

def read_attenuation_fl(angle_id, fpath_atten, elem):
    # fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_att = fpath_atten + f'/atten_fl_{elem}_prj_{angle_id:04d}.tiff'
    coef_att_fl = io.imread(fn_att)
    return coef_att_fl

def read_attenuation_xray(angle_id, fpath_atten, elem):
    # fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_att = fpath_atten + f'/atten_incident_xray_{elem}_prj_{angle_id:04d}.tiff'
    coef_att_xray = io.imread(fn_att)
    return coef_att_xray

def read_projection(angle_id, fpath_atten, elem):
    fn_prj = fpath_atten + f'/{elem}_ref_prj_{angle_id:04d}.tiff'
    img_prj = io.imread(fn_prj)
    return img_prj


def read_attenuation_at_all_angle(angle_list, fpath_atten, elem, mode='all'):
    '''
    mode:
    "all": read total attenuation from xrf and incident x-ray
    "fl": read fluorescent atten
    "xray": read incident x-ray attenuation
    '''
    print('reading attenuation files ...')
    n = len(angle_list)
    for i in trange(n):
        if mode == 'fl':
            if i == 0:
                print('reading fl')
            tmp = read_attenuation_fl(i, fpath_atten, elem)
        elif mode == 'xray':
            tmp = read_attenuation_xray(i, fpath_atten, elem)
            if i == 0:
                print('reading xray')
        else:
            tmp = read_attenuation(i, fpath_atten, elem)
            if i == 0:
                print('reading full atten')
        if i == 0:
            s = tmp.shape
            coef_att4D = np.zeros((n, *s))
        coef_att4D[i] = tmp
    return coef_att4D


############################
#### write attenunation ####
############################
def write_attenuation(elem, data, angle_id, file_path='./Angle_prj'):

    """
    write attenuation coefficient into file in the directory of './Angle_prj' as:

    'atten_gd_prj_-45.h5'

    where:  elem = 'gd'
            current_angle = -45

    """

    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root  = file_path
    else:
        fn_root  = file_path + '/'
    fname = fn_root + 'atten_' + elem + '_prj_' + f'{angle_id:04d}' + '.tiff'
    io.imsave(fname, data.astype(np.float32))

def write_attenuation_fl(elem, data, angle_id, file_path='./Angle_prj'):

    """
    write fluorescence attenuation coefficient for element:

    'atten_gd_prj_-45.h5'

    where:  elem = 'gd'
            current_angle = -45

    """

    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root  = file_path
    else:
        fn_root  = file_path + '/'
    fname = fn_root + 'atten_fl_' + elem + '_prj_' + f'{angle_id:04d}' + '.tiff'
    io.imsave(fname, data.astype(np.float32))


def write_attenuation_xray(elem, data, angle_id, file_path='./Angle_prj'):
    """
    write incident x-ray attenuation coefficient for element:

    'atten_gd_prj_0000.h5'

    where:  elem = 'gd'
            angle_id = 0
    """

    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root  = file_path
    else:
        fn_root  = file_path + '/'
    fname = fn_root + 'atten_incident_xray_' + elem + '_prj_' + f'{angle_id:04d}' + '.tiff'
    io.imsave(fname, data.astype(np.float32))


def write_projection(mode, elem, data, current_angle, angle_id, file_path='./Angle_prj'):
    """
    write aligned projection image into file with filename as:

    'gd_ref_prj_-45.h5'

    where:  elem = 'gd'
            current_angle = -45

    Parematers:
    -----------

    mode: chars, choosing from:
        "single_file" or "s" --> save as single projection file, then 'current_angle' is a anlge_list
        "multi_file" or "m"  --> save as multi (seperated) projection file, then 'current_angle' is simply a number

    data: 2D or 3D, depending on write into seperated projection file(2D), or a single combined file(3D)
    """
    mk_directory(file_path)
    if file_path[-1] == '/':
        fn_root = file_path
    else:
        fn_root = file_path + '/'
    if mode == 'single_file' or mode == 's':
        fname = fn_root + elem + '_ref_prj_single_file.h5'
        with h5py.File(fname, 'w') as hf:
            hf.create_dataset('dataset_1', data=data)
            hf.create_dataset('angle_list', data=current_angle)
    elif mode == 'multi_file' or mode == 'm':
        fname = fn_root + elem + '_ref_prj_' + f'{angle_id:04d}' + '.tiff'
        io.imsave(fname, data.astype(np.float32))
    else: print ('unrecognized "mode"')

############################
#### read/write recon ######
############################

def load_recon(fn_recon):
    if fn_recon[-1] == '/':
        fn_recon = fn_recon[:-1]
    fn_list = np.sort(glob.glob(fn_recon + '/*.tiff'))
    recon = {}
    ele_group = []
    keys = []
    for fn in fn_list:
        fn_short = fn.split('/')[-1]
        tmp = fn_short.split('_')
        ele = tmp[0]
        it = int(tmp[-1].split('.')[0])
        img = io.imread(fn)
        if it not in recon.keys():
            recon[it] = {}
        recon[it][ele] = img
        ele_group.append(ele)
        keys.append(it)
    ele_unique = list(set(ele_group))
    key_unique = list(set(keys))
    n_iter = np.max(key_unique)
    print(f'File loaded: recon{key_unique} \nkeys found: {ele_unique}')
    return recon, n_iter


def save_recon(rec_dict, fn_root, idx=None, save_hdf=False):
    fsave_recon = fn_root + '/recon'
    #os.makedirs(fsave_recon, exist_ok=True)
    mk_directory(fsave_recon)
    iters = len(rec_dict)
    if idx is None:
        for idx in range(iters):
            for k in rec_dict[idx].keys():
                fs = fsave_recon + f'/{k}_iter_{idx:02d}.tiff'
                io.imsave(fs, rec_dict[idx][k].astype(np.float32))
    else:
        for k in rec_dict[idx].keys():
            fs = fsave_recon + f'/{k}_iter_{idx:02d}.tiff'
            io.imsave(fs, rec_dict[idx][k].astype(np.float32))
    if save_hdf:
        fsave_hdf = fsave_recon + '/recon.h5'
        save_dict_to_hdf5(rec_dict, fsave_hdf)


def read_recon_elem(fpath_recon, iter_id, elem):
    fn = fpath_recon + f'/{elem}_iter_{iter_id:02d}.tiff'
    tmp = io.imread(fn)
    return tmp


def inspect_recon(recon_raw, iter_id, fn_root, elem, angle_list, ang_idx=0, sli=None):
    fpath_atten = fn_root + f'/Angle_prj_{iter_id:02d}'
    ang = angle_list[ang_idx]
    n = len(angle_list)
    s_tomo = recon_raw.shape
    img_raw_rot = rot3D(recon_raw, ang)
    recon_correction = read_recon_elem(fn_root + '/recon', iter_id, elem)
    img_rot = rot3D(recon_correction, ang)
    if sli is None:
        sli = recon_correction.shape[0] // 2

    coef_att = read_attenuation(ang_idx, fpath_atten, elem)
    img_prj = read_projection(ang_idx, fpath_atten, elem)

    img_att = (coef_att * img_rot)
    img_reprj = np.sum(img_att, axis=1)
    clim = [0, np.max(img_reprj) * 1.01]

    plt.figure(figsize=(18, 8))
    plt.suptitle(f'{elem} (iter={iter_id}): comparison at slice={sli}')

    plt.subplot(2, 3, 1);
    plt.imshow(img_raw_rot[sli], cmap='turbo');
    plt.colorbar()
    plt.title(f'tomo: raw reconstruction, angle={ang}')

    plt.subplot(2, 3, 2);
    plt.imshow(img_prj, clim=clim, cmap='twilight_r');
    plt.colorbar();
    plt.title(f'proj: measured (raw), angle={ang} ')

    plt.subplot(2, 3, 4);
    plt.imshow(img_rot[sli], cmap='turbo');
    plt.colorbar()
    plt.title(f'tomo: corrected, angle={ang}')

    plt.subplot(2, 3, 5);
    plt.imshow(img_reprj, clim=clim, cmap='twilight_r');
    plt.colorbar();
    plt.title(f'proj: attenu-simulated, angle={ang}')

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

    plt.subplot(2, 3, 3);
    plt.imshow(sino_raw, cmap='viridis');
    plt.colorbar()
    plt.title(f'sino: measured (raw)')

    plt.subplot(2, 3, 6);
    plt.imshow(sino_cor, cmap='viridis')
    plt.colorbar();
    plt.title(f'sino: corrected')
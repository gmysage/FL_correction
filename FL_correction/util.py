from skimage import io

#from examples.HXN_sparse_tomo.data_process_2025Q1 import init_guess
from .image_util import *
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import glob
import xraylib
from scipy.optimize import least_squares, curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from numpy.polynomial.polynomial import polyfit, polyval


def find_nearest(data, x):
    tmp = np.abs(data - x)
    return np.argmin(tmp)


def mk_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def exclude_idx(data, exclude_index):
    n1 = len(data)
    data_r = []
    for i in range(n1):
        if i in exclude_index:
            continue
        data_r.append(data[i])
    data_r = np.array(data_r)
    return data_r


def interp_line(x0, y0, x_interp, k=1):
    f = InterpolatedUnivariateSpline(x0, y0, k=3)
    y_interp = f(x_interp)
    return y_interp

def load_recon(fn_recon, id_iter='all'):
    if fn_recon[-1] == '/':
        fn_recon = fn_recon[:-1]
    fn_list = np.sort(glob.glob(fn_recon + '/*'))
    recon = {}
    ele_group = []
    keys = []
    for fn in fn_list:
        fn_short = fn.split('/')[-1]
        tmp = fn_short.split('_')
        ele = tmp[1]
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
    for it in key_unique:
        if 'sum' in recon[it].keys():
            continue
        recon[it]['sum'] = 0
        for ele in ele_unique:
            try:
                int(ele[-1]) # if it is Ni2, it will bypass it
                continue
            except:
                recon[it]['sum'] += recon[it][ele]
    if id_iter == 'all':
        return recon, n_iter
    else:
        rec = {}
        try:
            rec[id_iter] = recon[id_iter]
            return rec, id_iter
        except Exception as err:
            print(err)
            print('will return the whole dataset of recon')
            return recon, n_iter



def read_attenuation(angle_id, fpath_atten, elem):
    #fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_att = fpath_atten + f'/atten_{elem}_prj_{angle_id:04d}.tiff'
    coef_att = io.imread(fn_att)
    return coef_att

def read_attenuation_fl(angle_id, fpath_atten, elem):
    #fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_att = fpath_atten + f'/atten_fl_{elem}_prj_{angle_id:04d}.tiff'
    coef_att_fl = io.imread(fn_att)
    return coef_att_fl


def read_attenuation_xray(angle_id, fpath_atten, elem):
    #fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
    fn_att = fpath_atten + f'/atten_incident_xray_{elem}_prj_{angle_id:04d}.tiff'
    coef_att_xray = io.imread(fn_att)
    return coef_att_xray


def read_projection(angle_id, fpath_atten, elem):
    #fpath_atten = fn_root + f'/Angle_prj_{iter_id}'
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
            if i==0:
                print('reading fl')
            tmp = read_attenuation_fl(i, fpath_atten, elem)
        elif mode == 'xray':
            tmp = read_attenuation_xray(i, fpath_atten, elem)
            if i==0:
                print('reading xray')
        else:
            tmp = read_attenuation(i, fpath_atten, elem)
            if i==0:
                print('reading full atten')
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
    img_rot = rot3D(recon_correction, ang)
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


def exclude_eng(x_eng, y_spec, eng_exclude=[8.33, 8.4]):
    x = x_eng.copy()
    y = y_spec.copy()
    if len(eng_exclude) == 2:
        xs, xe = eng_exclude
        ids = find_nearest(x_eng, xs)
        ide = find_nearest(x_eng, xe)
        x = np.array(list(x_eng[:ids]) + list(x_eng[ide:]))
        y = np.array(list(y_spec[:ids]) + list(y_spec[ide:]))
    return x, y

def compose_compound(comp, ratio):
    # e.g. comp = ['LiNiO2', 'LiCoO2', 'LiMnO2']
    n = len(comp)
    c = ''
    r = np.array(ratio)
    r = r / np.sum(r)
    for i in range(n):
        if r[i] > 0:
            c = c + f'({comp[i]}){r[i]}'
    return c

def compose_cs(compound, x_eng):
    n = len(x_eng)
    tmp_cs = np.zeros(n)
    for i in range(n):
        tmp_cs[i] = xraylib.CS_Total_CP(compound, x_eng[i])
    return tmp_cs


def extract_em_cs_with_reference(param, elem, use_ref=True):
    em_cs = param['em_cs']

    eng_list = param['eng_list']
    n_eng = len(eng_list)
    if use_ref:
        n_ref = em_cs['n_ref']
        try:
            em_cs_ref = np.zeros((n_eng, n_ref))
            for i in range(n_ref):
                em_cs_ref[:, i] = em_cs[f'{elem}_ref{i+1}']
        except:
            em_cs_ref = em_cs[elem]
    else:
        em_cs_ref = em_cs[elem]
    return em_cs_ref

def spectrum_residual_composition_thickness(init_guess, x_data, y_data, composition_list, rho):
    # assume there are 3 component in the compound
    # e.g., (LiNiO2){xx}(LiCoO2){yy}(LiMnO2){zz}
    # param = [xx, yy, thickness], and zz = 1 - xx - yy
    # rho: mass density, e.g., for NMC, rho=4.65
    # x_data: energy list
    # y_data: spectrum after -log()
    # composition_list = e.g., ['LiNiO2', 'LiCoO2', 'LiMnO2']
    xx, yy, thick = init_guess
    zz = 1 - xx - yy
    xx = np.round(xx, 3)
    yy = np.round(yy, 3)
    zz = np.round(zz, 3)
    xx = min(1, xx)
    yy = min(1, yy)
    zz = min(1, zz)
    thick = max(thick, 1e-6)
    ratio = [xx, yy, zz]
    compound = compose_compound(composition_list, ratio)
    tmp_cs = compose_cs(compound, x_data)
    y = tmp_cs * rho * thick
    y_dif = y_data - y
    return y_dif

def fit_cs_thickness(composition_list, init_guess, x_eng, y_spec, rho, eng_exclude=[], plot_flag=1):
    # init_guess = [compositino_ratio1, composition_ratio2, thickness(cm)], e.g., [0.8, 0.1, 5e-4]
    x1, y1 = exclude_eng(x_eng, y_spec, eng_exclude)
    result = least_squares(spectrum_residual_composition_thickness, init_guess, args=(x1, y1, composition_list, rho))
    c1, c2, thick = result.x
    c3 = 1 - c1 - c2
    compound = compose_compound(composition_list, [c1, c2, c3])
    tmp_cs = compose_cs(compound, x_eng)
    y_fit = tmp_cs * rho * thick
    fit_res = {}
    fit_res['composition'] = [c1, c2, c3]
    fit_res['compound'] = compound
    fit_res['thick'] = thick
    fit_res['cs'] = tmp_cs
    fit_res['x'] = x_eng
    fit_res['y'] = y_fit
    if plot_flag:
        plt.figure()
        plt.plot(x_eng, y_spec, alpha=0.3, label='spectrum')
        plt.plot(x1, y1, '.', alpha=0.8, label='selected for fit')
        plt.plot(x_eng, y_fit, label='fitted')
        plt.title(f'{compound}\n thickness = {np.round(thick * 1e4, 2)} um')
        plt.legend()

    return fit_res

def update_incident_xray_cs_with_ref(cs_raw, ref_spec, elem='Ni', ref_ratio=[], plot_flag=True):
    # ref_spec: [ref1[eng, spec], ref2[eng, spec]]
    cs_with_ref = cs_raw.copy()
    new_keys = []
    x_incident_eng = cs_raw['x']
    keys = cs_raw.keys()
    for k_elem in keys:
        if '-x' in k_elem and elem in k_elem:
            break
    # e.g., k_elem = 'LiNiO2-x'
    k_comp = k_elem[:-2]
    atom_idx = xraylib.SymbolToAtomicNumber(elem)
    k_edge = xraylib.EdgeEnergy(atom_idx, 0) # 0 means k-edge

    n = len(ref_spec)
    if len(ref_ratio) != n:
        ref_ratio = np.ones(n) / n
    cs_with_ref[k_elem] = 0  # need to update cs['LiNiO2-x']

    simu_eng = np.linspace(k_edge-0.5, k_edge+0.5, 100)
    simu_cs = np.zeros(100)
    for i in range(100):
        simu_cs[i] = xraylib.CS_Total_CP(k_comp, simu_eng[i])
    pre_edge = [simu_eng[0], simu_eng[45]]
    post_edge = [simu_eng[55], simu_eng[-1]]
    cs_fit_simu, cs_pre_simu, cs_post_simu = normalize_1D_xanes(simu_cs, simu_eng, pre_edge, post_edge) # get slope of pre- and post-edge

    for i in range(n):
        ref_label = f'{k_elem}_ref{i+1}'  # LiNiO2-x_ref1, LiNiO2-x_ref2
        simu_label = f'simu_{k_elem}_ref{i+1}' # simu_LiNiO2-x_ref1, simu_LiNiO2-x_ref2
        simu_label_eng = simu_label + '_eng'
        x = ref_spec[i][:, 0]
        y = ref_spec[i][:, 1]

        cs_pre_ref = interp_line(simu_eng, cs_pre_simu, x)
        cs_post_ref = interp_line(simu_eng, cs_post_simu, x)

        cs_with_ref[simu_label] = y * cs_post_ref + cs_pre_ref
        # interpolate the for incident-x-ray-energy
        cs_with_ref[ref_label] = interp_line(x, cs_with_ref[simu_label], x_incident_eng, k=1)

        # update the 'LiNiO2-x' by ratio assigned
        cs_with_ref[k_elem] += cs_with_ref[ref_label] * ref_ratio[i]
        new_keys.append(simu_label)
        new_keys.append(ref_label)

        cs_with_ref[simu_label_eng] = x
        new_keys.append(simu_label_eng)
    cs_with_ref['n_ref'] = n
    new_keys.append('n_ref')
    print(f'\nkeys are added:\n{np.sort(list(new_keys))}\n')
    print(f'\nkeys are updated:\n{k_elem}')
    if plot_flag:
        plt.figure()
        for i in range(n):
            ref_label = f'{k_elem}_ref{i + 1}'  # LiNiO2-x_ref1, LiNiO2-x_ref2
            simu_label = f'simu_{k_elem}_ref{i + 1}'  # LiNiO2-x_ref1_simu, LiNiO2-x_ref2_simu
            x = ref_spec[i][:, 0]
            y = ref_spec[i][:, 1]
            plt.plot(x, cs_with_ref[simu_label], alpha=0.5, label=f'ref_{i+1}')
            plt.plot(x_incident_eng, cs_with_ref[ref_label], '.', label=f'ref_{i+1}')
        plt.plot(simu_eng, simu_cs, label=f'Edge jump: {elem}')
        plt.legend()
        plt.title('updated absorption cross-section')
    return cs_with_ref, simu_eng, simu_cs


def update_emission_xray_cs_with_ref(em_cs_raw, ref_spec, elem='Ni', ref_ratio=[], plot_flag=True):
    # ref_spec: [ref1[eng, spec], ref2[eng, spec]]
    em_cs_with_ref = em_cs_raw.copy()
    new_keys = []
    x_incident_eng = em_cs_raw['x']

    atom_idx = xraylib.SymbolToAtomicNumber(elem)
    k_edge = xraylib.EdgeEnergy(atom_idx, 0) # 0 means k-edge

    n = len(ref_spec)
    if len(ref_ratio) != n:
        ref_ratio = np.ones(n) / n
    em_cs_with_ref[elem] = 0  # need to update cs['LiNiO2-x']

    simu_eng = np.linspace(k_edge-0.5, k_edge+0.5, 100)
    simu_cs = np.zeros(100)
    for i in range(100):
        try:
            simu_cs[i] = xraylib.CS_FluorLine(atom_idx, 0, simu_eng[i])
        except:
            simu_cs[i] = 0
    pre_edge = [simu_eng[0], simu_eng[45]]
    post_edge = [simu_eng[55], simu_eng[-1]]
    cs_fit_simu, cs_pre_simu, cs_post_simu = normalize_1D_xanes(simu_cs, simu_eng, pre_edge, post_edge) # get slope of pre- and post-edge

    for i in range(n):
        ref_label = f'{elem}_ref{i+1}'  # Ni-x_ref1, Ni-x_ref2
        simu_label = f'simu_{elem}_ref{i+1}' # simu_Ni-x_ref1_simu, simu_Ni-x_ref2
        simu_label_eng = simu_label + '_eng'
        x = ref_spec[i][:, 0]
        y = ref_spec[i][:, 1]

        cs_pre_ref = interp_line(simu_eng, cs_pre_simu, x)
        cs_post_ref = interp_line(simu_eng, cs_post_simu, x)

        em_cs_with_ref[simu_label] = y * cs_post_ref + cs_pre_ref

        em_cs_with_ref[ref_label] = interp_line(x, em_cs_with_ref[simu_label], x_incident_eng, k=1)

        # update the 'Ni-x' by ratio assigned
        em_cs_with_ref[elem] += em_cs_with_ref[ref_label] * ref_ratio[i]
        new_keys.append(simu_label)
        new_keys.append(ref_label)
        em_cs_with_ref[simu_label_eng] = x
        new_keys.append(simu_label_eng)
    em_cs_with_ref['n_ref'] = n
    new_keys.append('n_ref')
    print(f'\nkeys are added:\n{np.sort(list(new_keys))}\n')
    print(f'\nkeys are updated:\n{elem}')
    if plot_flag:
        plt.figure()
        for i in range(n):
            ref_label = f'{elem}_ref{i + 1}'  # Ni-x_ref1, LiNiO2-x_ref2
            simu_label = f'simu_{elem}_ref{i + 1}'  # simu_LiNiO2-x_ref1, simu_LiNiO2-x_ref2
            x = ref_spec[i][:, 0]
            y = ref_spec[i][:, 1]
            plt.plot(x, em_cs_with_ref[simu_label], alpha=0.5, label=f'ref_{i+1}')
            plt.plot(x_incident_eng, em_cs_with_ref[ref_label], '.', label=f'ref_{i+1}')
        plt.plot(simu_eng, simu_cs, label=f'Edge jump: {elem}')
        plt.legend()
        plt.title('updated emission cross-section')
    return em_cs_with_ref, simu_eng, simu_cs


def cal_rho_compound(img4D, param):
    '''
    img4D contains 3D volume of all elements defined in param['elem_compound']
    '''
    elem_comp = param['elem_compound']
    nelem = len(elem_comp)
    rho = param['rho']
    rho_comp = 0
    img_sum = np.sum(img4D, axis=0)
    s = img_sum.shape
    frac = np.zeros((nelem, *s))
    m = np.zeros(s)
    m[img_sum>0] = 1
    for i in range(nelem):
        frac[i] = img4D[i] / img_sum * m
    frac[np.isnan(frac)] = 0
    frac[np.isinf(frac)] = 0
    for i in range(nelem):
        rho_comp += rho[elem_comp[i]] * frac[i]
    return rho_comp

def gaussian_function(x, amplitude, x_mean, sigma, offset):
    return amplitude * np.exp(-(x-x_mean)**2 / (2 * sigma**2)) + offset


def fit_gauss(x, y, x_mean=None):
    if len(x) - len(y) == 1:
        x = (x[1:] + x[:-1]) / 2
    if len(y) - len(x) == 1:
        y = (y[1:] + y[:-1]) / 2
    if x_mean is None:
        t = x[y>0.005*np.max(y)]
        x_mean = np.mean(t)
    else:
        x_mean = np.mean(x)
    init_guess = [np.max(y), np.mean(t), x_mean, np.min(y)*0.1]
    optimized_params, covariance_matrix = curve_fit(
        gaussian_function,
        x,
        y,
        p0=init_guess,
        maxfev=10000
    )
    amplitude, mean, sigma, offset = optimized_params
    xx = np.linspace(np.min(x), np.max(x), 1000)
    yy = gaussian_function(xx, *optimized_params)
    fwhm = np.abs(2.355 * sigma)
    return xx, yy, fwhm

##############################################
## following function are copied from pyxas
##############################################
def normalize_1D_xanes(xanes_spec, xanes_eng, pre_edge, post_edge):
    pre_s, pre_e = pre_edge
    post_s, post_e = post_edge
    x_eng = xanes_eng
    xanes_spec_fit = xanes_spec.copy()
    xs, xe = find_nearest(x_eng, pre_s), find_nearest(x_eng, pre_e)
    pre_eng = x_eng[xs:xe]
    pre_spec = xanes_spec[xs:xe]
    if len(pre_eng) > 1:
        y_pre_fit = fit_curve(pre_eng, pre_spec, x_eng, 1)
        xanes_spec_tmp = xanes_spec - y_pre_fit
        pre_fit_flag = True
    elif len(pre_eng) <= 1:
        y_pre_fit = np.ones(x_eng.shape) * xanes_spec[xs]
        xanes_spec_tmp = xanes_spec - y_pre_fit
        pre_fit_flag = True
    else:
        print('invalid pre-edge assignment')

    # fit post-edge
    xs, xe = find_nearest(x_eng, post_s), find_nearest(x_eng, post_e)
    post_eng = x_eng[xs:xe]
    post_spec = xanes_spec_tmp[xs:xe]
    if len(post_eng) > 1:
        y_post_fit = fit_curve(post_eng, post_spec, x_eng, 1)
        post_fit_flag = True
    elif len(post_eng) <= 1:
        y_post_fit = np.ones(x_eng.shape) * xanes_spec_tmp[xs]
        post_fit_flag = True
    else:
        print('invalid pre-edge assignment')

    if pre_fit_flag and post_fit_flag:
        xanes_spec_fit = xanes_spec_tmp * 1.0 / y_post_fit
        xanes_spec_fit[np.isnan(xanes_spec_fit)] = 0
        xanes_spec_fit[np.isinf(xanes_spec_fit)] = 0

    return xanes_spec_fit, y_pre_fit, y_post_fit

def fit_curve(x_raw, y_raw, x_fit, deg=1):
    coef1 = polyfit(x_raw, y_raw, deg)
    y_fit = polyval(x_fit, coef1)
    return y_fit
from skimage import io

# from examples.HXN_sparse_tomo.data_process_2025Q1 import init_guess
from .image_util import *
import numpy as np
import os
import h5py
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.signal import savgol_filter


def save_dict_to_hdf5(my_dict, fsave):
    with h5py.File(fsave, 'w') as f:
        _save_dict(my_dict, f)

def load_dict_to_hdf5(fn_dict):
    with h5py.File(fn_dict, "r") as f:
        my_dict = _load_hdf5_to_dict(f)
    return my_dict

def _save_dict(d, hdf5_file, parent_group=None):
    """
    Recursively save a dictionary to an HDF5 file.

    :param d: Dictionary to save
    :param hdf5_file: HDF5 file object (opened in write mode)
    :param parent_group: Group object to nest the dictionary under (default is None, which is the root group)
    """
    if parent_group is None:
        parent_group = hdf5_file  # Use the root group if no parent is provided

    for key, value in d.items():
        # If the value is a dictionary, create a new group and recurse
        if isinstance(value, dict):
            group = parent_group.create_group(key)  # Create a new group for this key
            _save_dict(value, hdf5_file, parent_group=group)  # Recurse into the dictionary
        else:
            # Otherwise, create a dataset for the value
            parent_group.create_dataset(key, data=value)


def _load_hdf5_to_dict(hdf5_obj):
    """
    Recursively load an HDF5 group or file into a dictionary.

    :param hdf5_obj: h5py.File or h5py.Group
    :return: Nested dictionary
    """
    result = {}

    for key, item in hdf5_obj.items():
        if isinstance(item, h5py.Group):
            # Recurse into groups
            result[key] = _load_hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            # Read dataset value
            data = item[()]
            # Optional: convert NumPy scalars to Python types
            if isinstance(data, np.generic):
                data = data.item()
            result[key] = data
    return result

def find_nearest(data, x):
    tmp = np.abs(data - x)
    return np.argmin(tmp)


def mk_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def pre_treat(*args):

    """
    Combine a list of 3D data(with same shape) to a 4D array.

    Parameters:
    -----------
    a list of 3D data, or one 4D array

    Returns:
    --------
    4D array, and a array consist of the shape of 3D data

    """

    n = len(args)
    if n == 1:                                    # one set of 4D or 3D data
        data = np.squeeze(np.array(args))
        if len(data.shape) == 4:
            data = rm_nan(data)
            return np.array(data)
        elif len(data.shape) == 3:
            data = np.zeros([1]+list(data.shape))
            data[0] = args[0]
            data = rm_nan(data)
            return np.array(data)
        else:
            print('input should be either 3D or 4D data')
            return 0
    if n > 1:                                     # a list of 3D or 2D data
        s = args[0].shape
        ss = [n] + [s[i] for i in range(len(s))]
        data = np.zeros(ss)
        for i in range(n):
            data[i] = args[i]
        data = rm_nan(data)
        return np.array(data)


def exclude_idx(data, exclude_index):
    n1 = len(data)
    data_r = []
    for i in range(n1):
        if i in exclude_index:
            continue
        data_r.append(data[i])
    data_r = np.array(data_r)
    return data_r


def smooth_savgol(y, window_length=11, polyorder=3):
    y = np.asarray(y)
    # Ensure window_length is valid
    if window_length % 2 == 0:
        raise ValueError("window_length must be an odd integer.")
    if window_length <= polyorder:
        raise ValueError("window_length must be greater than polyorder.")
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)


def fit_peak_curve_spline(x, y, fit_order=3, smooth=0.002, weight=[1]):
    if not len(weight) == len(x):
        weight = np.ones((len(x)))
    spl = UnivariateSpline(x, y, k=fit_order, s=smooth, w=weight)
    xx = np.linspace(x[0], x[-1], 10001)
    yy = spl(xx)
    peak_pos = xx[np.argmax(yy)]
    fit_error = np.sum((y - spl(x) ** 2))
    edge_pos = xx[np.argmax(np.abs(np.diff(spl(xx))))]
    res = {}
    res['peak_pos'] = peak_pos
    res['peak_val'] = spl(peak_pos)
    res['edge_pos'] = edge_pos
    res['edge_val'] = spl(edge_pos)
    res['fit_error'] = fit_error
    res['spl'] = spl
    res['xx'] = xx
    return res


def interp_line(x0, y0, x_interp, k=1):
    f = InterpolatedUnivariateSpline(x0, y0, k=3)
    y_interp = f(x_interp)
    return y_interp


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
    m[img_sum > 0] = 1
    for i in range(nelem):
        frac[i] = img4D[i] / img_sum * m
    frac[np.isnan(frac)] = 0
    frac[np.isinf(frac)] = 0
    for i in range(nelem):
        rho_comp += rho[elem_comp[i]] * frac[i]
    return rho_comp


def gaussian_function(x, amplitude, x_mean, sigma, offset):
    return amplitude * np.exp(-(x - x_mean) ** 2 / (2 * sigma ** 2)) + offset


def fit_gauss(x, y, x_mean=None):
    if len(x) - len(y) == 1:
        x = (x[1:] + x[:-1]) / 2
    if len(y) - len(x) == 1:
        y = (y[1:] + y[:-1]) / 2
    if x_mean is None:
        t = x[y > 0.002 * np.max(y)]
        x_mean = np.mean(t)
    else:
        x_mean = np.mean(x)
    init_guess = [np.max(y), np.mean(t), x_mean, np.min(y) * 0.1]
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
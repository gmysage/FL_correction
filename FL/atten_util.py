from tqdm import trange
from copy import deepcopy
from .file_io import (read_attenuation_at_all_angle,
                      write_attenuation_xray,
                      write_attenuation_fl,
                      write_attenuation
                      )
from .image_util import rot3D
import glob
import numpy as np
import xraylib


def get_atten_coef(elem_type, elem_compound, XEng_list, em_E):
    '''
    elem_type = ['Zr', 'La', 'Hf']
    elem_compound = ['ZrO2', 'La2O3', 'HfO2']
    XEng = [12]
    em_E = [4.6, ]
    '''
    et = elem_type
    ec = elem_compound
    cs = {}
    n = len(elem_type)
    cs['elem_type'] = elem_type
    cs['elem_compound'] = elem_compound
    cs['x'] = XEng_list # incident x-ray energy
    for i in range(n):
        # atten at incident x-ray
        cs[f'{ec[i]}-x'] = np.zeros(len(XEng_list))
        for k in range(len(XEng_list)):
            XEng = XEng_list[k]
            # cs[f'{et[i]}-x'][k] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(et[i]), XEng)
            cs[f'{ec[i]}-x'][k]  = xraylib.CS_Total_CP(ec[i], XEng)
        for j in range(n):
            # atten at each emission line
            # cs[f'{et[i]}-{et[j]}'] = xraylib.CS_Total(xraylib.SymbolToAtomicNumber(et[i]), em_E[et[j]])
            cs[f'{ec[i]}-{et[j]}'] = xraylib.CS_Total_CP(ec[i], em_E[et[j]])
    return cs

def cal_total_atten_at_angles(fsave, param):
    elem_type = param['elem_type']
    n_ele = len(elem_type)
    n_angle = int(len(glob.glob(fsave+ '/atten_fl*'))/n_ele)

    for elem in elem_type:
        att_ele_xrf = read_attenuation_at_all_angle(np.arange(n_angle),
                                                    fsave,
                                                    elem,
                                                    'fl')
                                                        
        att_ele_xray = read_attenuation_at_all_angle(np.arange(n_angle),
                                                     fsave,
                                                     elem,
                                                     'xray')
        att = att_ele_xrf * att_ele_xray
        for i in range(n_angle):
            write_attenuation(elem, att[i], i, fsave)

###################################
# help functions
###################################
def extract_em_cs_elem(param, elem, use_ref=False):
    em_cs_dict = param['em_cs']
    em_cs = em_cs_dict[elem]
    n = len(em_cs)
    if use_ref:
        try:
            n_ref = em_cs_dict['n_ref']
            em_cs = np.zeros((n, n_ref))
            for i in range(n_ref):
                em_cs[:, i] = em_cs_dict[f'{elem}_ref{i+1}']
        except Exception as err:
            print(err)
            print(f'fail to extract em_cs for {elem} using reference')
    return em_cs


def update_dict_use_ref_at_angle(dict_use_ref, current_angle):
    dict_use_ref_angle = deepcopy(dict_use_ref)
    # need to rotate dict_use_ref['ref_3D']
    if len(dict_use_ref_angle) > 0:
        if dict_use_ref_angle['use_ref']:
            ref_3D = dict_use_ref_angle['ref_3D']
            n_ref = len(ref_3D)
            for i in range(n_ref):
                dict_use_ref_angle['ref_3D'][i] = rot3D(ref_3D[i], current_angle)
    return dict_use_ref_angle

def extract_single_value_param(param, energy_angle_id, dict_use_ref={}):
    elem_type = param['elem_type']
    elem_comp = param['elem_compound']
    param_angle_energy = param.copy()
    cs = param['cs']
    cs_angle_energy = deepcopy(cs)
    cs_angle_energy_type = type(cs_angle_energy[f'{elem_comp[0]}-x'])
    if cs_angle_energy_type is np.ndarray or list:
        for ele_comp in elem_comp:
            if len(cs[f'{ele_comp}-x']) == 1:
                cs_angle_energy[f'{ele_comp}-x'] = cs[f'{ele_comp}-x']
            else:
                cs_angle_energy[f'{ele_comp}-x'] = cs[f'{ele_comp}-x'][energy_angle_id]

    em_cs = param['em_cs']
    em_cs_angle_energy = deepcopy(em_cs)
    em_cs_angle_energy_type = type(em_cs_angle_energy[f'{elem_type[0]}'])

    if em_cs_angle_energy_type is np.ndarray or list:
        for ele in elem_type:
            if len(em_cs[ele]) == 1:
                em_cs_angle_energy[ele] = em_cs[ele]
            else:
                em_cs_angle_energy[ele] = em_cs[ele][energy_angle_id]

    """
    if reference spectrum provided, will compose the absorption cs with reference and chemical 
    concentration, e.g., dict_use_ref['ref_3D'] = (Ni2, Ni3), 
    Here, we only pick up the cs of reference at individual energy. 
    Note that id_angle == id_energy

    It pass the 'dict_use_ref' to function 'cal_atten_3D' to do real calculation 
    """
    if len(dict_use_ref) > 0:
        if dict_use_ref['use_ref']:  # True of False
            # update absorption cross-section
            for k in cs_angle_energy.keys():
                if 'x_ref' in k:
                    cs_angle_energy[k] = cs[k][energy_angle_id]
            # update emission cross-section
            for k in em_cs_angle_energy.keys():
                if ('x_ref' in k) and (not 'n_ref' in k):
                    em_cs_angle_energy[k] = em_cs[k][energy_angle_id]

    param_angle_energy['cs'] = cs_angle_energy
    param_angle_energy['em_cs'] = em_cs_angle_energy
    return param_angle_energy
import xraylib
import numpy as np
import matplotlib.pyplot as plt
from .util import interp_line, normalize_1D_xanes, exclude_eng
from scipy.optimize import least_squares

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
                em_cs_ref[:, i] = em_cs[f'{elem}_ref{i + 1}']
        except:
            em_cs_ref = em_cs[elem]
    else:
        em_cs_ref = em_cs[elem]
    return em_cs_ref


def compose_cs_incident_xray_with_reference(ref_3D, ref_cs, ref_comp):
    '''
    ref_3D = (Ni2, Ni3), e.g., Ni2(3).shape=(100, 100, 100)
    ref_cs.keys = ['LiNiO2-x_ref1', 'LiNiO2-x_ref2', ...]
    ref_comp = 'LiNiO2'

    ref_cs: e.g., ['LiNiO2-x_ref1', 'LiNiO2-x_ref2']
        ref_cs['LiNiO2-x_ref1']: a single value, corresponding to the reference cross-section at single x-ray energy
        Note: 'ref_cs' is adapted from "cs_angle_energy" from function "cal_and_save_atten_prj"

    return:
    cs_4D: 3D cross-section at all energies, shape = (n_eng, 100, 100, 100)
    '''
    n_ref = ref_cs['n_ref']
    assert len(ref_3D) == n_ref, "number of references and number of 3D stack in 'ref_3D' does not match"
    ref_spec = np.zeros(n_ref)
    for i_ref in range(n_ref):
        ref_spec[i_ref] = ref_cs[f'{ref_comp}-x_ref{i_ref + 1}']
    ref_frac = cal_frac(ref_3D, scale_range=[], enable_scale=False)['frac']
    cs_3D = 0
    for i_ref in range(n_ref):
        cs_3D += ref_spec[i_ref] * ref_frac[i_ref]
    return cs_3D


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
    k_edge = xraylib.EdgeEnergy(atom_idx, 0)  # 0 means k-edge

    n = len(ref_spec)
    if len(ref_ratio) != n:
        ref_ratio = np.ones(n) / n
    cs_with_ref[k_elem] = 0  # need to update cs['LiNiO2-x']

    simu_eng = np.linspace(k_edge - 0.5, k_edge + 0.5, 100)
    simu_cs = np.zeros(100)
    for i in range(100):
        simu_cs[i] = xraylib.CS_Total_CP(k_comp, simu_eng[i])
    pre_edge = [simu_eng[0], simu_eng[45]]
    post_edge = [simu_eng[55], simu_eng[-1]]
    cs_fit_simu, cs_pre_simu, cs_post_simu = normalize_1D_xanes(simu_cs, simu_eng, pre_edge,
                                                                post_edge)  # get slope of pre- and post-edge

    for i in range(n):
        ref_label = f'{k_elem}_ref{i + 1}'  # LiNiO2-x_ref1, LiNiO2-x_ref2
        simu_label = f'simu_{k_elem}_ref{i + 1}'  # simu_LiNiO2-x_ref1, simu_LiNiO2-x_ref2
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
            plt.plot(x, cs_with_ref[simu_label], alpha=0.5, label=f'ref_{i + 1}')
            plt.plot(x_incident_eng, cs_with_ref[ref_label], '.', label=f'ref_{i + 1}')
        plt.plot(simu_eng, simu_cs, label=f'Edge jump: {elem}')
        plt.legend()
        plt.title('updated absorption cross-section')
    return cs_with_ref, simu_eng, simu_cs

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

    # composition_list = ['LiNiO2', 'LiCoO2', 'LiMnO2']
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
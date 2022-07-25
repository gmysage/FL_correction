import numpy as np
import tomopy
import numpy as np
from skimage import io
import glob
from tqdm import trange
import FLCorrection_new as FL
import h5py

global mask3D

def cal_atten_prj_at_angle(angle, img4D, param, cs, position_det='r', num_cpu=8):
    '''
    calculate the attenuation and projection at single angle
    '''
    elem_type = cs['elem_type']
    s = img4D.shape
    n_type = len(elem_type)
    img4D_r = FL.rot3D(img4D, angle)
    prj = {}
    atten = FL.cal_atten_with_direction(img4D_r, cs, param, mask3D, position_det='r', num_cpu=num_cpu)
    for j in range(n_type):
        ele = elem_type[j]
        prj[ele] = np.sum(img4D_r[j]*atten[ele], axis=1)
    res = {}
    res['atten'] = atten
    res['prj'] = prj
    return res


def simu_atten_prj(angle_list, img4D, param, cs, position_det='r', file_path='.', num_cpu=8):
    elem_type = cs['elem_type']
    s = img4D.shape
    n_type = s[0]
    n = len(angle_list)
    prj = {}
    for j in range(n_type):
        ele = elem_type[j]
        prj[ele] = np.zeros((n, s[1], s[3]))
    for i in range(n):
        print(f'\nsimu attenuated projection at angle {angle_list[i]}: {i+1}/{n}')
        res = cal_atten_prj_at_angle(angle_list[i], img4D, param, cs, position_det='r', num_cpu=num_cpu)
        for j in range(n_type):
            elem = elem_type[j]
            prj[elem][i] = res['prj'][elem]
            FL.write_attenuation(elem, res['atten'][elem], angle_list[i], file_path)
            FL.write_projection('m', elem, prj[elem][i], angle_list[i], file_path)
    return prj


def simu_projection(simu, simu_size, mu_elem, angle_list, PixSize):
    '''
    simulating the absorbed projection image from a 3D-phantom (composed of Gd-Fe)
    Specifically, incident X-ray travels from "top->down", detector locates at "left"
    '''

    # initialize projection image for two elements at all angles
    prj = np.zeros([Nelem, angle_list.size, simu_size[0], simu_size[2]])

    for ii in range(Nelem):
        for ang in range(angle_list.size):
            simu_rot = FL.rot3D(simu, angle_list[ang])
            mu_elem_rot = FL.rot3D(mu_elem, angle_list[ang])

            # calculating attenuation coef. of incident x-ray shining from 'left->right'
            att_incident_xray = FL.atten_incident_xray('td', Att_Xray, PixSize, simu_rot)

            # calculating attenuation coef. for element fluorescent X-ray
            print(f'Calculating attenuation for element #{ii+1} at angle {angle_list[ang]}')

            att_elem = FL.atten_fluorescence('left', mu_elem_rot[ii], PixSize, pad_thick=0)
            att_total = att_elem #* att_incident_xray

            temp3D = simu_rot[ii] * att_total
            prj[ii, ang] = np.sum(temp3D, 1)
    return prj


def simu_write_projection(prj, elem_list, angle_list, file_path='phantom_FL_test/simulated_proj'):
    if np.size(elem_list) == 1:
        FL.write_projection(mode='s', elem=elem_list, data=prj, current_angle=angle_list, file_path=file_path)
    else:
        for i in range(np.size(elem_list)):
            FL.write_projection('s', elem_list[i], prj[i], angle_list, file_path=file_path)


def simu_tomography(proj, angle_list, sli=[], num_iter=20):
    '''
    simulating the tomography reconstruction using tomopy mlem algorithm
    '''
    prj = FL.pre_treat(proj)
    rot_center = np.array(prj.shape[3])/2.0-0.5
    if np.max(np.abs(angle_list))>np.pi:
        the = np.array(angle_list)/180*np.pi
    else:
        the = angle_list
    elem_num = prj.shape[0]
    rec = np.zeros([elem_num, prj.shape[2], prj.shape[3], prj.shape[3]])
    for i in range(elem_num):
        if sli == []:
            data = prj[i]
        else:
            data = prj[i, :, sli:sli+1, :]
        rec[i] = tomopy.recon(data, the, rot_center, num_iter=num_iter, algorithm='mlem')
    return rec


def cal_and_save_atten_prj(param, recon4D, angle_list, ref_prj, fsave='./Angle_prj', align_flag=0, num_cpu=8):
    '''
    Function used to calcuate the attenuation coefficents of each elements and all 3D voxels
    results will be saved into the "fsave" folder

    If align_flag = 1:
        it will do image alignment using the forwared projection image as reference, and align
    the raw projection image ("ref_prj": the projection image collected from experiment)

    If align_flag = 0:
        it will simply copy the raw projection image into the folder, which assuming the raw projection images
        is well alinged with rotation center locted at image center.

    '''
    s = recon4D.shape
    elem_type = param['elem_type']
    Nelem = len(elem_type)
    n_angle = len(angle_list)
    prj = np.zeros([Nelem, n_angle, s[1], s[3]])
    for ang in range(n_angle):
        print(f'\ncalculate and save attenuation and projection at angle {angle_list[ang]}: {ang+1}/{n_angle}')
        res = cal_atten_prj_at_angle(angle_list[ang], recon4D, param, cs, position_det='r', num_cpu=num_cpu)
        for i in range(Nelem):
            elem = elem_type[i]
            if align_flag:
                prj[i, ang],_,_ = FL.align_img(res['prj'][elem], ref_prj[i, ang])
            else:
                prj[i, ang] = ref_prj[i, ang]
            FL.write_projection('m', elem, prj[i, ang], angle_list[ang], fsave)
            FL.write_attenuation(elem, res['atten'][elem], angle_list[ang], fsave)


def simu_absorption_correction(sli, elem, ref_tomo, angle_list, file_path, iter_num=10):
    I_tot = FL.generate_I(elem, ref_tomo, sli, angle_list, bad_angle_index=[], file_path=file_path)
    H = FL.generate_H(elem, ref_tomo, sli, angle_list, bad_angle_index=[], file_path=file_path, flag=1)
    img_cor = FL.mlem_matrix(ref_tomo[sli], H, I_tot, iter_num=iter_num)
    return img_cor

def simu_absorption_correction_mpi(sli, elem, ref_tomo, angle_list, file_path, iter_num):
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm
    from functools import partial
    partial_func = partial(simu_absorption_correction, elem=elem,
                        ref_tomo=ref_tomo, angle_list=angle_list,
                        file_path=file_path, iter_num=iter_num)
    pool = Pool(8)
    res = []
    ts = time.time()
    for result in tqdm(pool.imap(func=partial_func, iterable=sli), total=len(sli)):
        res.append(result)
    pool.close()
    pool.join()
    te = time.time()
    print(f'take {te-ts:2.1f} sec')
    return np.array(res)


def prep_detector_mask3D(alfa=15, theta=60, length_maximum=200):
    print('Generating detector 3D mask ...')
    mask = {}
    for i in trange(length_maximum+1, 7, -1):
        mask[f'{i}'] = FL.generate_detector_mask(alfa, theta, i)
    with h5py.File('mask3D.h5', 'w') as hf:
        for i in range(7, length_maximum+1):
            k = f'{i}'
            hf.create_dataset(k, data=mask[k])

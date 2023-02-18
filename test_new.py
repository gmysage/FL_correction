import tomopy
import numpy as np
import h5py
from skimage import io
import FLCorrection_new as FL
import matplotlib.pylab as plt
import glob
from tqdm import trange
from numba import jit, njit, prange
from utils import *
global mask3D

"""
General information for this script

In the simulation, it starts with two 3D objections contains Zr and Hf.
step 0: generate a detection mask, related to the detection angle of the XRF detector. 
        Save this mask for further use.

Step 1: load related 3D files (ground truth) and other parameter (defined in param_Zr_Hf.txt)

Step 2 (optional):  generate ground truth projection images without self-absorption. For comparison purpose

Step 3: generate the attenuated projection image at each angle. 
        It simulates the projection image collected at XRF beamline
        (this step takes long time, ~ few hours)

Step 4: tomography reconstruction based on the attenuated projection image. 
        This is the starting point for absorption correction

Step 5: iterative correction. 
        (this step takes long time, ~ few hours)


"""


def main():
    # (sli, row, col)
    # if we look at the a 2D slice (reconstructed):
    # x-ray direction: from bottom to top
    # detector: right side of the 2D slice image

    #### step 0:
    #### prepare 3D mask, representing the solid angle of detector ###

    # in the demonstration, we assum the detector is rectangular shape
    # acceptance angle in horizontal direaction is set to 15 degrees
    # acceptance angle in vertical direaction is set to 60 degrees



    fn = './'

    # the calculation will be slow, be patient !
    # usually run once, and save it to local disk for future use
    mask3D = prep_detector_mask3D(alfa=15, theta=60, length_maximum=200) # alfa: horizontal angle; theta: vertical angle; length_maximum: set as image size

    ####################################################################

    #### in future, just load the mask file: ##################
    # mask3D = FL.load_mask3D(fn +'mask3D.h5')
    ###########################################################


    

    #### step 1: load parameters
    Zr = io.imread(fn + 'Zr3D_00.tiff')[20:80]
    Hf = io.imread(fn + 'Hf3D_00.tiff')[20:80]
    img4D = FL.pre_treat(Zr, Hf)
    param = FL.load_param(fn + 'param_Zr_Hf.txt')
    cs = FL.get_atten_coef(param['elem_type'], param['XEng'], param['em_E'])
    elem_type = param['elem_type']
    angle_list = np.arange(0, 180, 3)
    num_cpu = 4


    #### step 2 (optional): generate perfect tomo reconstruction for reference
    fn_ground_truth = fn + f'ground_truth/'
    mk_directory(fn_ground_truth)

    Zr_prj_perfect = FL.re_projection(Zr, angle_list)
    Zr_rec_perfect = tomopy.recon(Zr_prj_perfect, angle_list/180*np.pi,
                    Zr_prj_perfect.shape[2]/2-0.5,algorithm='mlem', num_iter=20)
    Zr_rec_perfect = Zr_rec_perfect[:,::-1]

    io.imsave(fn_ground_truth + 'prj_Zr_perfect.tiff', Zr_prj_perfect.astype(np.float32))
    io.imsave(fn_ground_truth + 'rec_Zr_perfect.tiff', Zr_rec_perfect.astype(np.float32))

    Hf_prj_perfect = FL.re_projection(Hf, angle_list)
    Hf_rec_perfect = tomopy.recon(Hf_prj_perfect, angle_list/180*np.pi,
                    Hf_prj_perfect.shape[2]/2-0.5,algorithm='mlem', num_iter=20)
    Hf_rec_perfect = Hf_rec_perfect[:,::-1]
    io.imsave(fn_ground_truth + 'prj_Hf_perfect.tiff', Hf_prj_perfect.astype(np.float32))
    io.imsave(fn_ground_truth + 'rec_Hf_perfect.tiff', Hf_rec_perfect.astype(np.float32))


    #### step 3:
    # SIMULATING the attenuated projection image at all angles
    # Equivalent to experimental projection image at all angles

    # will save attenuation and projection into file
    # Be patient, this will take some time
    fsave = fn+'Angle_prj_ground_truth/'
    mk_directory(fsave)

    prj_atten = simu_atten_prj(angle_list, img4D, param, cs, mask3D, position_det='r', file_path=fsave, num_cpu=num_cpu)

    # save projection image
    for i in range(len(elem_type)):
        elem = elem_type[i]
        FL.write_projection('s', elem, prj_atten[elem], angle_list, file_path=fn+'simulated_proj')

    # read simulated projection
    f1 = h5py.File(fn+'simulated_proj/Zr_ref_prj_single_file.h5', 'r')
    f2 = h5py.File(fn+'simulated_proj/Hf_ref_prj_single_file.h5', 'r')
    Zr_prj = np.array(f1['dataset_1'])
    Hf_prj = np.array(f2['dataset_1'])
    f1.close()
    f2.close()

    # Step 4
    # blind reconstruction using simulated-attenuated projection as what we got from experiment
    # this serves as the starting point for the followinng correction
    s = Zr_prj.shape
    simu_prj = FL.pre_treat(Zr_prj, Hf_prj)
    simu_rec = simu_tomography(simu_prj, angle_list)
    simu_rec = simu_rec[:, :, ::-1, :]


# step 5: start correction
#### 1st iteration
    # calculate the attenuation based on the current reconstruction
    cal_and_save_atten_prj(param, cs, simu_rec, angle_list, simu_prj, mask3D, fsave=fn+'Angle_prj1', align_flag=0, num_cpu=num_cpu)
    sli = np.arange(100)
    m = (Zr>0).astype(np.int16)
    ref_tomo = np.ones(Zr.shape) * m

    # correction on Zr
    Zr_cor1 = simu_absorption_correction_mpi(sli, elem_type[0], ref_tomo, angle_list=angle_list, file_path=fn+'Angle_prj1', iter_num=30)
    io.imsave(fn+'Zr_cor_01.tiff', Zr_cor1.astype(np.float32))

    # correction on Hf
    Hf_cor1 = simu_absorption_correction_mpi(sli, elem_type[1], ref_tomo, angle_list=angle_list, file_path=fn+'Angle_prj1', iter_num=30)
    io.imsave(fn+'Hf_cor_01.tiff', Hf_cor1.astype(np.float32))

#### 2st iteration
    m = (Zr>0).astype(np.int16)
    ref_tomo = np.ones(Zr.shape) * m
    Zr_cor1 = io.imread(fn+'Zr_cor_01.tiff')
    Hf_cor1 = io.imread(fn+'Hf_cor_01.tiff')
    simu_rec1 = FL.pre_treat(Zr_cor1, Hf_cor1)
    cal_and_save_atten_prj(param, cs, simu_rec1, angle_list, simu_prj, mask3D, fsave=fn+'Angle_prj2', align_flag=0, num_cpu=num_cpu)
    sli = np.arange(100)
    Zr_cor2 = simu_absorption_correction_mpi(sli, elem_type[0], ref_tomo, angle_list=angle_list, file_path=fn+'Angle_prj2', iter_num=30)
    io.imsave(fn+'Zr_cor_02.tiff', Zr_cor2.astype(np.float32))
    Hf_cor2 = simu_absorption_correction_mpi(sli, elem_type[1], ref_tomo, angle_list=angle_list, file_path=fn+'Angle_prj2', iter_num=30)
    io.imsave(fn+'Hf_cor_02.tiff', Hf_cor2.astype(np.float32))


#### summary
    # Zr and Hf are the ground_truth
    # simu_rec[0] and simu_rec[1] are the blind-reconstructions of Zr and Hf using attenuated projection image, which simulated the projection collected from experiment
    # Zr_cor1 and Hf_cor1 are the results after 2 iterations.
    # Zr_cor2 and Hf_cor2 are the results after 2 iterations.

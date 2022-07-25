import numpy as np
import scipy.fftpack as sf
import math
import matplotlib.pyplot as mplp
import scipy.ndimage as sn


def dftregistration(buf1ft,buf2ft,usfac=100):
   """
       # function [output Greg] = dftregistration(buf1ft,buf2ft,usfac);
       # Efficient subpixel image registration by crosscorrelation. This code
       # gives the same precision as the FFT upsampled cross correlation in a
       # small fraction of the computation time and with reduced memory
       # requirements. It obtains an initial estimate of the
crosscorrelation peak
       # by an FFT and then refines the shift estimation by upsampling the DFT
       # only in a small neighborhood of that estimate by means of a
       # matrix-multiply DFT. With this procedure all the image points
are used to
       # compute the upsampled crosscorrelation.
       # Manuel Guizar - Dec 13, 2007

       # Portions of this code were taken from code written by Ann M. Kowalczyk
       # and James R. Fienup.
       # J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
       # object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
       # (1990).

       # Citation for this algorithm:
       # Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
       # "Efficient subpixel image registration algorithms," Opt. Lett. 33,
       # 156-158 (2008).

       # Inputs
       # buf1ft    Fourier transform of reference image,
       #           DC in (1,1)   [DO NOT FFTSHIFT]
       # buf2ft    Fourier transform of image to register,
       #           DC in (1,1) [DO NOT FFTSHIFT]
       # usfac     Upsampling factor (integer). Images will be registered to
       #           within 1/usfac of a pixel. For example usfac = 20 means the
       #           images will be registered within 1/20 of a pixel.
(default = 1)

       # Outputs
       # output =  [error,diffphase,net_row_shift,net_col_shift]
       # error     Translation invariant normalized RMS error between f and g
       # diffphase     Global phase difference between the two images (should be
       #               zero if images are non-negative).
       # net_row_shift net_col_shift   Pixel shifts between images
       # Greg      (Optional) Fourier transform of registered version of buf2ft,
       #           the global phase difference is compensated for.
   """

   # Compute error for no pixel shift
   if usfac == 0:
       CCmax = np.sum(buf1ft*np.conj(buf2ft))
       rfzero = np.sum(abs(buf1ft)**2)
       rgzero = np.sum(abs(buf2ft)**2)
       error = 1.0 - CCmax*np.conj(CCmax)/(rgzero*rfzero)
       error = np.sqrt(np.abs(error))
       diffphase = np.arctan2(np.imag(CCmax),np.real(CCmax))
       return error, diffphase

   # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
   # peak
   elif usfac == 1:
       ndim = np.shape(buf1ft)
       m = ndim[0]
       n = ndim[1]
       CC = sf.ifft2(buf1ft*np.conj(buf2ft))
       max1,loc1 = idxmax(CC)
       rloc = loc1[0]
       cloc = loc1[1]
       CCmax=CC[rloc,cloc]
       rfzero = np.sum(np.abs(buf1ft)**2)/(m*n)
       rgzero = np.sum(np.abs(buf2ft)**2)/(m*n)
       error = 1.0 - CCmax*np.conj(CCmax)/(rgzero*rfzero)
       error = np.sqrt(np.abs(error))
       diffphase=np.arctan2(np.imag(CCmax),np.real(CCmax))
       md2 = np.fix(m/2)
       nd2 = np.fix(n/2)
       if rloc > md2:
           row_shift = rloc - m
       else:
           row_shift = rloc

       if cloc > nd2:
           col_shift = cloc - n
       else:
           col_shift = cloc

       ndim = np.shape(buf2ft)
       nr = int(round(ndim[0]))
       nc = int(round(ndim[1]))
       Nr = sf.ifftshift(np.arange(-np.fix(1.*nr/2),np.ceil(1.*nr/2)))
       Nc = sf.ifftshift(np.arange(-np.fix(1.*nc/2),np.ceil(1.*nc/2)))
       Nc,Nr = np.meshgrid(Nc,Nr)
       Greg = buf2ft*np.exp(1j*2*np.pi*(-1.*row_shift*Nr/nr-1.*col_shift*Nc/nc))
       Greg = Greg*np.exp(1j*diffphase)
       image_reg = sf.ifft2(Greg) * np.sqrt(nr*nc)

       #return error,diffphase,row_shift,col_shift
       return error,diffphase,row_shift,col_shift, image_reg

   # Partial-pixel shift
   else:

       # First upsample by a factor of 2 to obtain initial estimate
       # Embed Fourier data in a 2x larger array
       ndim = np.shape(buf1ft)
       m = int(round(ndim[0]))
       n = int(round(ndim[1]))
       mlarge=m*2
       nlarge=n*2
       CC=np.zeros([mlarge,nlarge],dtype=np.complex128)

       CC[int(m-np.fix(m/2)):int(m+1+np.fix((m-1)/2)),int(n-np.fix(n/2)):int(n+1+np.fix((n-1)/2))] = (sf.fftshift(buf1ft)*np.conj(sf.fftshift(buf2ft)))[:,:]


       # Compute crosscorrelation and locate the peak
       CC = sf.ifft2(sf.ifftshift(CC)) # Calculate cross-correlation
       max1,loc1 = idxmax(np.abs(CC))

       rloc = int(round(loc1[0]))
       cloc = int(round(loc1[1]))
       CCmax = CC[rloc,cloc]

       # Obtain shift in original pixel grid from the position of the
       # crosscorrelation peak
       ndim = np.shape(CC)
       m = ndim[0]
       n = ndim[1]

       md2 = np.fix(m/2)
       nd2 = np.fix(n/2)
       if rloc > md2:
           row_shift = rloc - m
       else:
           row_shift = rloc

       if cloc > nd2:
           col_shift = cloc - n
       else:
           col_shift = cloc

       row_shift=row_shift/2
       col_shift=col_shift/2

       # If upsampling > 2, then refine estimate with matrix multiply DFT
       if usfac > 2:
           ### DFT computation ###
           # Initial shift estimate in upsampled grid
           row_shift = 1.*np.round(row_shift*usfac)/usfac;
           col_shift = 1.*np.round(col_shift*usfac)/usfac;
           dftshift = np.fix(np.ceil(usfac*1.5)/2); ## Center of output array at dftshift+1
           # Matrix multiply DFT around the current shift estimate
           CC = np.conj(dftups(buf2ft*np.conj(buf1ft),np.ceil(usfac*1.5),np.ceil(usfac*1.5),usfac,\
               dftshift-row_shift*usfac,dftshift-col_shift*usfac))/(md2*nd2*usfac**2)
           # Locate maximum and map back to original pixel grid
           max1,loc1 = idxmax(np.abs(CC))
           rloc = int(round(loc1[0]))
           cloc = int(round(loc1[1]))

           CCmax = CC[rloc,cloc]
           rg00 = dftups(buf1ft*np.conj(buf1ft),1,1,usfac)/(md2*nd2*usfac**2)
           rf00 = dftups(buf2ft*np.conj(buf2ft),1,1,usfac)/(md2*nd2*usfac**2)
           rloc = rloc - dftshift
           cloc = cloc - dftshift
           row_shift = 1.*row_shift + 1.*rloc/usfac
           col_shift = 1.*col_shift + 1.*cloc/usfac

       # If upsampling = 2, no additional pixel shift refinement
       else:
           rg00 = np.sum(buf1ft*np.conj(buf1ft))/m/n;
           rf00 = np.sum(buf2ft*np.conj(buf2ft))/m/n;

       error = 1.0 - CCmax*np.conj(CCmax)/(rg00*rf00);
       error = np.sqrt(np.abs(error));
       diffphase = np.arctan2(np.imag(CCmax),np.real(CCmax));
       # If its only one row or column the shift along that dimension has no
       # effect. We set to zero.
       if md2 == 1:
          row_shift = 0

       if nd2 == 1:
          col_shift = 0;

       # Compute registered version of buf2ft
       if usfac > 0:
          ndim = np.shape(buf2ft)
          nr = ndim[0]
          nc = ndim[1]
          Nr = sf.ifftshift(np.arange(-np.fix(1.*nr/2),np.ceil(1.*nr/2)))
          Nc = sf.ifftshift(np.arange(-np.fix(1.*nc/2),np.ceil(1.*nc/2)))
          Nc,Nr = np.meshgrid(Nc,Nr)
          Greg = buf2ft*np.exp(1j*2*np.pi*(-1.*row_shift*Nr/nr-1.*col_shift*Nc/nc))
          Greg = Greg*np.exp(1j*diffphase)
       elif (nargout > 1) and (usfac == 0):
          Greg = np.dot(buf2ft,np.exp(1j*diffphase))
          
       #plt.figure(3)
       image_reg = sf.ifft2(Greg) * np.sqrt(nr*nc)
       #imgplot = plt.imshow(np.abs(image_reg))

       #a_ini = np.zeros((100,100))
       #a_ini[40:59,40:59] = 1.
       #a = a_ini * np.exp(1j*15.) 
       #plt.figure(6)
       #imgplot = plt.imshow(np.abs(a))
       #plt.figure(3)
       #imgplot = plt.imshow(np.abs(a)-np.abs(image_reg))
       #plt.colorbar()

       # return error,diffphase,row_shift,col_shift,Greg
       return error,diffphase,row_shift,col_shift, image_reg


def dftups(inp,nor,noc,usfac=1,roff=0,coff=0):
   """
       # function out=dftups(in,nor,noc,usfac,roff,coff);
       # Upsampled DFT by matrix multiplies, can compute an upsampled
DFT in just
       # a small region.
       # usfac         Upsampling factor (default usfac = 1)
       # [nor,noc]     Number of pixels in the output upsampled DFT, in
       #               units of upsampled pixels (default = size(in))
       # roff, coff    Row and column offsets, allow to shift the
output array to
       #               a region of interest on the DFT (default = 0)
       # Recieves DC in upper left corner, image center must be in (1,1)
       # Manuel Guizar - Dec 13, 2007
       # Modified from dftus, by J.R. Fienup 7/31/06

       # This code is intended to provide the same result as if the following
       # operations were performed
       #   - Embed the array "in" in an array that is usfac times larger in each
       #     dimension. ifftshift to bring the center of the image to (1,1).
       #   - Take the FFT of the larger array
       #   - Extract an [nor, noc] region of the result. Starting with the
       #     [roff+1 coff+1] element.

       # It achieves this result by computing the DFT in the output
array without
       # the need to zeropad. Much faster and memory efficient than the
       # zero-padded FFT approach if [nor noc] are much smaller than
[nr*usfac nc*usfac]
   """

   ndim = np.shape(inp)
   nr = int(round(ndim[0]))
   nc = int(round(ndim[1]))
   noc = int(round(noc))
   nor = int(round(nor))

   # Compute kernels and obtain DFT by matrix products
   a = np.zeros([nc,1])
   a[:,0] = ((sf.ifftshift(np.arange(nc)))-np.floor(1.*nc/2))[:]
   b = np.zeros([1,noc])
   b[0,:] = (np.arange(noc)-coff)[:]
   kernc = np.exp((-1j*2*np.pi/(nc*usfac))*np.dot(a,b))
   nndim = kernc.shape
   #print nndim

   a = np.zeros([nor,1])
   a[:,0] = (np.arange(nor)-roff)[:]
   b = np.zeros([1,nr])
   b[0,:] = (sf.ifftshift(np.arange(nr))-np.floor(1.*nr/2))[:]
   kernr = np.exp((-1j*2*np.pi/(nr*usfac))*np.dot(a,b))
   nndim = kernr.shape
   #print nndim

   return np.dot(np.dot(kernr,inp),kernc)



def idxmax(data):
   ndim = np.shape(data)
   #maxd = np.max(data)
   maxd = np.max(np.abs(data))
   #t1 = plt.mlab.find(np.abs(data) == maxd)
   t1 = np.argmin(np.abs((np.abs(data) - maxd)))
   idx = np.zeros([len(ndim),])
   for ii in range(len(ndim)-1):
       t1,t2 = np.modf(1.*t1/np.prod(ndim[(ii+1):]))
       idx[ii] = t2
       t1 *= np.prod(ndim[(ii+1):])
   idx[np.size(ndim)-1] = t1

   return maxd,idx


def flip_conj(tmp):
    #ndims = np.shape(tmp)
    #nx = ndims[0]
    #ny = ndims[1]
    #nz = ndims[2]
    #tmp_twin = np.zeros([nx,ny,nz]).astype(complex)
    #for i in range(0,nx):
    #   for j in range(0,ny):
    #      for k in range(0,nz):
    #         i_tmp = nx - 1 - i
    #         j_tmp = ny - 1 - j
    #         k_tmp = nz - 1 - k
    #         tmp_twin[i,j,k] = tmp[i_tmp,j_tmp,k_tmp].conj()
    #return tmp_twin

    tmp_fft = sf.ifftshift(sf.ifftn(sf.fftshift(tmp)))
    return sf.ifftshift(sf.fftn(sf.fftshift(np.conj(tmp_fft)))) 

def check_conj(ref, tmp,threshold_flag, threshold,subpixel_flag):
    ndims = np.shape(ref)
    nx = ndims[0]
    ny = ndims[1]
    nz = ndims[2]

    if threshold_flag == 1:
       ref_tmp = np.zeros((nx,ny,nz))
       index = np.where(np.abs(ref) >= threshold*np.max(np.abs(ref)))
       ref_tmp[index] = 1.
       tmp_tmp = np.zeros((nx,ny,nz))
       index = np.where(np.abs(tmp) >= threshold*np.max(np.abs(tmp)))
       tmp_tmp[index] = 1.
       tmp_conj = flip_conj(tmp_tmp)
    else:
       ref_tmp = ref
       tmp_tmp = tmp
       tmp_conj = flip_conj(tmp)
       
    tmp_tmp = subpixel_align(ref_tmp,tmp_tmp,threshold_flag,threshold,subpixel_flag)
    tmp_conj = subpixel_align(ref_tmp,tmp_conj,threshold_flag,threshold,subpixel_flag)

    cc_1 = sf.ifftn(ref_tmp*np.conj(tmp_tmp))
    cc1 = np.max(cc_1.real)
    #cc1 = np.max(np.abs(cc_1))
    cc_2 = sf.ifftn(ref_tmp*np.conj(tmp_conj))
    cc2 = np.max(cc_2.real)
    #cc2 = np.max(np.abs(cc_2))
    print('{0}, {1}'.format(cc1, cc2))
    if cc1 > cc2:
        return 0
    else:
        return 1

def subpixel_align(ref,tmp,threshold_flag,threshold, subpixel_flag):
    ndims = np.shape(ref)
    if np.size(ndims) == 3:
       nx = ndims[0]
       ny = ndims[1]
       nz = ndims[2]

       if threshold_flag == 1:
          ref_tmp = np.zeros((nx,ny,nz))
          index = np.where(np.abs(ref) >= threshold*np.max(np.abs(ref)))
          ref_tmp[index] = 1.
          tmp_tmp = np.zeros((nx,ny,nz))
          index = np.where(np.abs(tmp) >= threshold*np.max(np.abs(tmp)))
          tmp_tmp[index] = 1.
          ref_fft = sf.ifftn(sf.fftshift(ref_tmp))
          tmp_fft = sf.ifftn(sf.fftshift(tmp_tmp))
          real_fft = sf.ifftn(sf.fftshift(tmp))
       else:
          ref_fft = sf.ifftn(sf.fftshift(ref))
          tmp_fft = sf.ifftn(sf.fftshift(tmp))

       nest = np.mgrid[0:nx,0:ny,0:nz]

       result = dftregistration(ref_fft[:,:,0],tmp_fft[:,:,0],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_1 = cl
       y_shift_1 = r
       result = dftregistration(ref_fft[:,:,nz-1],tmp_fft[:,:,nz-1],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_2 = cl
       y_shift_2 = r
    
       result = dftregistration(ref_fft[:,0,:],tmp_fft[:,0,:],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_3 = cl
       z_shift_1 = r
       result = dftregistration(ref_fft[:,ny-1,:],tmp_fft[:,ny-1,:],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift_4 = cl
       z_shift_2 = r
       
       result = dftregistration(ref_fft[0,:,:],tmp_fft[0,:,:],usfac=100)
       e, p, cl, r, array_shift = result
       y_shift_3 = cl
       z_shift_3 = r
       result = dftregistration(ref_fft[nx-1,:,:],tmp_fft[nx-1,:,:],usfac=100)
       e, p, cl, r, array_shift = result
       y_shift_4 = cl
       z_shift_4 = r


       if subpixel_flag == 1:
          x_shift = (x_shift_1 + x_shift_2 + x_shift_3 + x_shift_4)/4.
          y_shift = (y_shift_1 + y_shift_2 + y_shift_3 + y_shift_4)/4.
          z_shift = (z_shift_1 + z_shift_2 + z_shift_3 + z_shift_4)/4.
       else:
          x_shift = np.floor((x_shift_1 + x_shift_2 + x_shift_3 + x_shift_4)/4.+0.5)
          y_shift = np.floor((y_shift_1 + y_shift_2 + y_shift_3 + y_shift_4)/4.+0.5)
          z_shift = np.floor((z_shift_1 + z_shift_2 + z_shift_3 + z_shift_4)/4.+0.5)

       print('x, y, z shift: {0}, {1}, {2}'.format(x_shift, y_shift, z_shift))

       if threshold_flag == 1:
          tmp_fft_new = sf.ifftshift(real_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:,:]-ny/2.)/(ny)-z_shift*(nest[2,:,:,:]-nz/2.)/(nz)))
       else:
          tmp_fft_new = sf.ifftshift(tmp_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:,:]-ny/2.)/(ny)-z_shift*(nest[2,:,:,:]-nz/2.)/(nz)))

    if np.size(ndims) == 2:
       nx = ndims[0]
       ny = ndims[1]

       if threshold_flag == 1:
          ref_tmp = np.zeros((nx,ny))
          index = np.where(np.abs(ref) >= threshold*np.max(np.abs(ref)))
          ref_tmp[index] = 1.
          tmp_tmp = np.zeros((nx,ny))
          index = np.where(np.abs(tmp) >= threshold*np.max(np.abs(tmp)))
          tmp_tmp[index] = 1.
          
          ref_fft = sf.ifftn(sf.fftshift(ref_tmp))
          mp_fft = sf.ifftn(sf.fftshift(tmp_tmp))
          real_fft = sf.ifftn(sf.fftshift(tmp))
       else:
          ref_fft = sf.ifftn(sf.fftshift(ref))
          tmp_fft = sf.ifftn(sf.fftshift(tmp))

       nest = np.mgrid[0:nx,0:ny]

       result = dftregistration(ref_fft[:,:],tmp_fft[:,:],usfac=100)
       e, p, cl, r, array_shift = result
       x_shift = cl
       y_shift = r

       if subpixel_flag == 1:
          x_shift = x_shift
          y_shift = y_shift
       else:
          x_shift = np.floor(x_shift + 0.5)
          y_shift = np.floor(y_shift + 0.5)

       print ('x, y shift: {0}, {1}'.format(x_shift, y_shift))

       if threshold_flag == 1:
          tmp_fft_new = sf.ifftshift(real_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))
       else:
          tmp_fft_new = sf.ifftshift(tmp_fft) * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))

    return sf.ifftshift(sf.fftn(sf.fftshift(tmp_fft_new))),x_shift,y_shift

    
def remove_phase_ramp(tmp,threshold_flag, threshold,subpixel_flag):
   tmp_tmp,x_shift,y_shift = subpixel_align(sf.ifftshift(sf.ifftn(sf.fftshift(np.abs(tmp)))), sf.ifftshift(sf.ifftn(sf.fftshift(tmp))), threshold_flag, threshold,subpixel_flag) 
   tmp_new = sf.ifftshift(sf.fftn(sf.fftshift(tmp_tmp)))
   phase_tmp = np.angle(tmp_new)
   ph_offset = np.mean(phase_tmp[np.where(np.abs(tmp) >= threshold)])
   phase_tmp = np.angle(tmp_new) - ph_offset
   return np.abs(tmp)*np.exp(1j*phase_tmp)

def pixel_shift(array,x_shift,y_shift,z_shift):
    nx,ny,nz = np.shape(array)
    tmp = sf.ifftshift(sf.ifftn(sf.fftshift(array)))
    nest = np.mgrid[0:nx,0:ny,0:nz]
    tmp = tmp * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:,:]-ny/2.)/(ny)-z_shift*(nest[2,:,:,:]-nz/2.)/(nz)))
    return sf.ifftshift(sf.fftn(sf.fftshift(tmp)))

def pixel_shift_2d(array,x_shift,y_shift):
    nx,ny = np.shape(array)
    tmp = sf.ifftshift(sf.ifftn(sf.fftshift(array)))
    nest = np.mgrid[0:nx,0:ny]
    tmp = tmp * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))
    return sf.ifftshift(sf.fftn(sf.fftshift(tmp)))

def rm_phase_ramp_manual_2d(array,x_shift,y_shift):
    nx,ny = np.shape(array)
    nest = np.mgrid[0:nx,0:ny]
    tmp = array * np.exp(1j*2*np.pi*(-1.*x_shift*(nest[0,:,:]-nx/2.)/(nx)-y_shift*(nest[1,:,:]-ny/2.)/(ny)))
    return tmp

if (__name__ == '__main__'):
    pass

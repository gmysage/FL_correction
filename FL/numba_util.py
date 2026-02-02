from numba import cuda, float32
from math import *
import numpy as np
from tqdm import trange


@cuda.jit
def kernel_generate_H_single_angle(tomo3D_shape, theta_i, att, H):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2) 
    g = cuda.cg.this_grid()

    m, n = H.shape
    for i in range(iy, n, threads_per_grid_y):
        for j in range(ix, m, threads_per_grid_x):
            H[m, n] = 0
    g.sync()
    s = tomo3D_shape
    cx = (s[2]-1) / 2.0       # center of col
    cy = (s[1]-1) / 2.0       # center of row
    #H_tot = np.zeros([s[2]*num, s[2]*s[2]])

    #att = atten3D[i]
    T00 = cos(-theta_i)
    T01 = -sin(-theta_i)
    T10 = sin(-theta_i)
    T11 = cos(-theta_i)

    for col in range(ix, s[2], threads_per_grid_x):
        for row in range(iy, s[1], threads_per_grid_y):
            p = row
            q = col
            
            t0 = T00 * (p-cx) + T01 * (q-cy) + cx
            t1 = T10 * (p-cx) + T11 * (q-cy) + cy
            if ((t0 > s[1]-1) or (t0 <= 0) or (t1 > s[2]-1) or (t1 <= 0)):
                continue
                
            r_frac = t0 - floor(t0)
            c_frac = t1 - floor(t1)
            r_up = int(floor(t0))
            r_down = int(ceil(t0))
            c_left = int(floor(t1))
            c_right = int(ceil(t1))

            ul = r_up * s[2] + c_left
            ur = r_up * s[2] + c_right
            dl = r_down * s[2] + c_left
            dr = r_down * s[2] + c_right

            if (r_up >= 0 and c_left >=0):
                H[q, ul] = H[q, ul] + att[p, q] * (1-r_frac) * (1-c_frac)
            if (c_left >=0):
                H[q, dl] = H[q, dl] + att[p, q] * r_frac * (1-c_frac)
            if (r_up >= 0):
                H[q, ur] = H[q, ur] + att[p,q] * (1-r_frac) * c_frac
            H[q, dr] =  H[q, dr] + att[p, q] * r_frac * c_frac
    

def new_cuda_array(shape):
    arr = np.zeros(shape, dtype=np.float32)
    arr = cuda.to_device(arr)
    return arr


@cuda.jit(cache=True)
def kernel_matmul(mA, mB, mC):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)    
    n0, n1 = mC.shape    
    for i in range(iy, n0, threads_per_grid_y):
        for j in range(ix, n1, threads_per_grid_x):
            tmp = 0.
            for k in range(mA.shape[1]):
                tmp = tmp + mA[i, k] * mB[k, j]
            mC[i, j] = tmp


@cuda.jit(cache=True)
def kernel_low_limit_2D_cuda(data, threashold, val):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)   
    s = data.shape
    for i in range(iy, s[0], threads_per_grid_y):
        for j in range(ix, s[1], threads_per_grid_x):
            if data[i, j] < threashold:
                data[i, j] = val

@cuda.jit(cache=True)
def kernel_element_divide_2D(A, B, C):
    # C = A / B
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)  
    s = A.shape
    for i in range(iy, s[0], threads_per_grid_y):
        for j in range(ix, s[1], threads_per_grid_x):
            if B[i, j] == 0:
                C[i, j] = 0
            else:
                C[i, j] = A[i, j] / B[i, j]


@cuda.jit(cache=True)
def kernel_element_multiply_2D(A, B, C):
    # C = A * B
    ix, iy = cuda.grid(2)

    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)  
    s = A.shape
    for i in range(iy, s[0], threads_per_grid_y):
        for j in range(ix, s[1], threads_per_grid_x):
            C[i, j] = A[i, j] * B[i, j]



@cuda.jit(cache=True)
def kernel_matrix_T(A, B):
    ix, iy = cuda.grid(2)
    threads_per_grid_x, threads_per_grid_y = cuda.gridsize(2)
    s = A.shape
    for i in range(iy, s[0], threads_per_grid_y):
        for j in range(ix, s[1], threads_per_grid_x):
            B[j, i] = A[i, i]


def example_mlem_cuda(X, A, B, iter_num=10):
    '''
    cuda version of solving X from AX=B
    all the inputs are cuda.array

    A.shape = (m, n)
    X.shape = (n, 1)
    B.shape = (m, 1)
    Intermediate variable:
        AX = A @ X                             --> shape = (m, 1)
        A_sum = np.sum(A, axis=0)              --> shape = (1, n)
        B_AX =  B / AX (elementwise)           --> shape = (m, 1)
        B_AX_T = B_AX.T (transpose)            --> shape = (1, m)
        A_sum_new = B_AX_T @ A                 --> shpae = (1, n)    
        R = A_sum_new / A_sum  (elementwise)   --> shape = (1, n)
        R_T = R.T                              --> shape = (n, 1)  
        X_new = X * R_T (elementwise)          --> shape = (n, 1)
    '''

    blocks, threads = 256, 256

    m, n = A.shape
    '''
    AX = new_cuda_array((m, 1))
    A_sum = new_cuda_array((1, n))
    B_AX = new_cuda_array((m, 1))
    #B_AX_T = new_cuda_array((1, m))
    A_sum_new = new_cuda_array((1, n))
    
    R = new_cuda_array((1, n))
    #R_T = new_cuda_array((n, 1))
    X_new = new_cuda_array((n, 1))
    '''
    I_tot = I_tot.reshape((len(I_tot), 1))
    y = I_tot
    B = cuda.to_device(I_tot.astype(np.float32))
    
    m, n = A.shape
    AX = np.zeros((m, 1), dtype=np.float32);     
    AX = cuda.to_device(AX)
    A_sum = np.zeros((1, n), dtype=np.float32);     
    A_sum = cuda.to_device(A_sum)
    B_AX = np.zeros((m, 1), dtype=np.float32);     
    B_AX = cuda.to_device(B_AX)
    B_AX_T = np.zeros((1, m), dtype=np.float32);     
    B_AX_T = cuda.to_device(B_AX_T)
    A_sum_new = np.zeros((1, n), dtype=np.float32);     
    A_sum_new = cuda.to_device(A_sum_new)
    R = np.zeros((1, n), dtype=np.float32);     
    R = cuda.to_device(R)
    R_T = np.zeros((n, 1), dtype=np.float32);     
    R_T = cuda.to_device(R_T)
    X_new = np.zeros((n, 1), dtype=np.float32);     
    X_new = cuda.to_device(X_new)
    
    blocks, threads = 256, 256

    #threads_per_block_2d = (16, 16)  #  1024 threads total
    #blocks_per_grid_2d = (64, 64)

    kernel_row_sum_1d[128, 128](A, A_sum)
    cuda.synchronize()

    #kernel_low_limit_2[blocks_per_grid_2d, threads_per_block_2d](B, 0, 0)
    kernel_low_limit_1D[blocks, threads](B, 0., 0.)
    cuda.synchronize()
    
    for i in trange(iter_num):
        kernel_matmul_1d[blocks, threads](A, X, AX)
        cuda.synchronize()
        
        kernel_low_limit_1D[blocks, threads](AX, 1e-6, 1)
        cuda.synchronize()

        kernel_element_divide_1D[blocks, threads](B, AX, B_AX)
        cuda.synchronize()

        #_matrix_T[blocks_per_grid_2d, threads_per_block_2d](B_AX, B_AX_T)
        B_AX_T = B_AX.T

        #matmul[blocks_per_grid_2d, threads_per_block_2d](B_AX_T, A, A_sum_new)
        kernel_matmul_1d[blocks, threads](B_AX_T, A, A_sum_new)
        cuda.synchronize()

        #kernel_element_divide_2D[blocks_per_grid_2d, threads_per_block_2d](A_sum_new, A_sum, R)
        kernel_element_divide_1D[blocks, threads](A_sum_new, A_sum, R)
        cuda.synchronize()

        #kernel_matrix_T[blocks_per_grid_2d, threads_per_block_2d](R, R_T)
        R_T = R.T

        #kernel_element_multiply_2D[blocks_per_grid_2d, threads_per_block_2d](X, R_T, X_new)
        kernel_element_multiply_1D[blocks, threads](X, R_T, X_new)
        cuda.synchronize()
        
        X = X_new
    return X_new



@cuda.jit()
def kernel_mlem(X, A, B, AX, A_sum, B_AX, A_sum_new, R, X_new):
    '''
    cuda version of solving X from AX=B
    all the inputs are cuda.array

    A.shape = (m, n)
    X.shape = (n, 1) 
    B.shape = (m, 1) 
    Intermediate variable:
        A_sum = np.sum(A, axis=0)              --> shape = (1, n) 
        AX = A @ X                             --> shape = (m, 1) 
        B_AX =  B / AX (elementwise)           --> shape = (m, 1) 
        B_AX_T = B_AX.T (transpose)            --> shape = (1, m) 
        A_sum_new = B_AX_T @ A                 --> shpae = (1, n) 
        R = A_sum_new / A_sum  (elementwise)   --> shape = (1, n) 
        R_T = R.T                              --> shape = (n, 1) 
        X_new = X * R_T (elementwise)          --> shape = (n, 1) 
    '''
    ix = cuda.grid(1)
    threads_per_grid = cuda.gridsize(1)
    m, n = A.shape

    for i in range(ix, m, threads_per_grid):
        if B[i, 0] < 0:
            B[i, 0] = 0

    # A_sum = np.sum(A, axis=0)  (n)
    for j in range(ix, n, threads_per_grid):
        tmp = 0
        for k in range(m):
            tmp = tmp + A[k, j]
        A_sum[0, j] = tmp

    # AX = A @ X_new  (m)
    for i in range(ix, m, threads_per_grid):
        tmp = 0
        for k in range(n):
            tmp = tmp + A[i, k] * X[k, 0]
        AX[i, 0] = tmp

    # B_AX = B / AX   (m)
    for i in range(ix, m, threads_per_grid):        
        if AX[i, 0] < 1e-6:
            B_AX[i, 0] = 0
        else:
            B_AX[i, 0] = B[i, 0] / AX[i, 0]

    # A_sum_new = B_AX_T @ A  (n)
    for j in range(ix, n, threads_per_grid):
        tmp = 0
        for k in range(m):
            tmp = tmp + B_AX[k, 0] * A[k, j]
        A_sum_new[0, j] = tmp

    for j in range(ix, n, threads_per_grid):
        if A_sum[0, j] == 0:
            R[0, j] = 0            
        else:
            R[0, j] = A_sum_new[0, j] / A_sum[0, j]

    # X_new = X * A_sum_new / A_sum (n)
    for j in range(ix, n, threads_per_grid):
        X_new[j, 0] = R[0, j] * X[j, 0]

    for j in range(ix, n, threads_per_grid):
        X[j, 0] = X_new[j, 0]
    

def mlem_cuda(X, A, B, n_iter=10, blocks=256, threads=256):
    '''
    cuda version of solving X from AX=B
    all the inputs are cuda.array

    A.shape = (m, n)
    X.shape = (n, 1)
    B.shape = (m, 1)
    Intermediate variable:
        AX = A @ X                             --> shape = (m, 1)
        A_sum = np.sum(A, axis=0)              --> shape = (1, n)
        B_AX =  B / AX (elementwise)           --> shape = (m, 1)
        B_AX_T = B_AX.T (transpose)            --> shape = (1, m)
        A_sum_new = B_AX_T @ A                 --> shpae = (1, n)    
        R = A_sum_new / A_sum  (elementwise)   --> shape = (1, n)
        R_T = R.T                              --> shape = (n, 1)  
        X_new = X * R_T (elementwise)          --> shape = (n, 1)
    '''
    #blocks, threads = 512, 512
    m, n = A.shape
    AX = new_cuda_array((m, 1))
    A_sum = new_cuda_array((1, n))
    B_AX = new_cuda_array((m, 1))
    A_sum_new = new_cuda_array((1, n))    
    R = new_cuda_array((1, n))
    kernel_mlem_iter[blocks, threads](X, A, B, AX, A_sum, B_AX, A_sum_new, R, n_iter)
    cuda.synchronize()
    

###########

@cuda.jit
def kernel_zeros_2Darray(arr):
    ix, iy = cuda.grid(2)
    threads_per_grid_x,  threads_per_grid_y = cuda.gridsize(2)
    m, n = arr.shape
    for i in range(iy, m, threads_per_grid_y):
        for j in range(ix, n, threads_per_grid_x):
            arr[i, j] = 0

@cuda.jit
def kernel_zeros_3Darray(arr):
    ix, iy, iz = cuda.grid(3)
    threads_per_grid_x, threads_per_grid_y, threads_per_grid_z= cuda.gridsize(3)
    l, m, n = arr.shape
    for i in range(iz, l, threads_per_grid_z):
        for j in range(iy, m, threads_per_grid_y):
            for k in range(ix, n, threads_per_grid_x):
                arr[i, j, k] = 0

@cuda.jit()
def kernel_mlem_iter(X, A, B, AX, A_sum, B_AX, A_sum_new, R, n_iter):
    '''
    cuda version of solving X from AX=B
    all the inputs are cuda.array

    A.shape = (m, n)
    X.shape = (n, 1) 
    B.shape = (m, 1) 
    Intermediate variable:
        A_sum = np.sum(A, axis=0)              --> shape = (1, n) 
        AX = A @ X                             --> shape = (m, 1) 
        B_AX =  B / AX (elementwise)           --> shape = (m, 1) 
        B_AX_T = B_AX.T (transpose)            --> shape = (1, m) 
        A_sum_new = B_AX_T @ A                 --> shpae = (1, n) 
        R = A_sum_new / A_sum  (elementwise)   --> shape = (1, n) 
        R_T = R.T                              --> shape = (n, 1) 
        X_new = X * R_T (elementwise)          --> shape = (n, 1) 
    '''
    ix = cuda.grid(1)
    threads_per_grid = cuda.gridsize(1)
    g = cuda.cg.this_grid()
    m, n = A.shape

    for i in range(ix, m, threads_per_grid):
        if B[i, 0] < 0:
            B[i, 0] = 0
    g.sync()
    # A_sum = np.sum(A, axis=0)  (n)
    for j in range(ix, n, threads_per_grid):
        tmp = 0
        for k in range(m):
            tmp = tmp + A[k, j]
        A_sum[0, j] = tmp
    g.sync()

    # start iteration
    for iters in range(n_iter):
        # AX = A @ X_new  (m)
        # AX[AX<1e-6] = 1
        for i in range(ix, m, threads_per_grid):
            tmp = 0
            for k in range(n):
                tmp = tmp + A[i, k] * X[k, 0]
            if tmp < 1e-6:
                tmp = 1
            AX[i, 0] = tmp
        g.sync()
        # B_AX = B / AX   (m)
        for i in range(ix, m, threads_per_grid):        
            if AX[i, 0] < 1e-6:
                B_AX[i, 0] = 0
            else:
                B_AX[i, 0] = B[i, 0] / AX[i, 0]
        g.sync()
        # A_sum_new = B_AX_T @ A  (n)
        for j in range(ix, n, threads_per_grid):
            tmp = 0
            for k in range(m):
                tmp = tmp + B_AX[k, 0] * A[k, j]
            A_sum_new[0, j] = tmp
        g.sync()
        for j in range(ix, n, threads_per_grid):
            if A_sum[0, j] == 0:
                R[0, j] = 0            
            else:
                R[0, j] = A_sum_new[0, j] / A_sum[0, j]
        g.sync()
        # X_new = X * A_sum_new / A_sum (n)
        for j in range(ix, n, threads_per_grid):
            X[j, 0] = R[0, j] * X[j, 0]
        g.sync()
        #for j in range(ix, n, threads_per_grid):
        #    X[j, 0] = X_new[j, 0]
        #g.sync()

##############



@cuda.jit
def kernel_matmul_1d(A, B, C):
    '''
    A @ B = C
    A = (m, n) or (1, m)
    B = (n, 1) or (m, n)
    C = (m, 1) or (1, n)
    '''
    p, q = C.shape    
    ix = cuda.grid(1)
    threads_per_grid=cuda.gridsize(1)
    if q == 1:
        m, n = A.shape
        for i in range(ix, m, threads_per_grid):
            tmp = 0
            for k in range(n):
                tmp = tmp + A[i, k] * B[k, 0]
            C[i, 0] = tmp    
    elif p == 1:
        m, n = B.shape
        ix = cuda.grid(1)
        threads_per_grid=cuda.gridsize(1)
        for i in range(ix, n, threads_per_grid):
            tmp = 0
            for k in range(m):
                tmp = tmp + A[0, k] * B[k, i]
            C[0, i] = tmp


@cuda.jit
def kernel_row_sum_1d(A, A_sum):
    '''
    A = (m, n)
    A_sum = (1, n)
    '''
    m, n = A.shape
    ix = cuda.grid(1)
    threads_per_grid=cuda.gridsize(1)
    for i in range(ix, n, threads_per_grid):    
        tmp = 0
        for k in range(m):
            tmp = tmp + A[k, i]
        A_sum[0, i] = tmp


@cuda.jit
def kernel_low_limit_1D(data, threshold, val):
    ix = cuda.grid(1)
    threads_per_grid=cuda.gridsize(1)
    m, n = data.shape
    if n == 1:
        for i in range(ix, m, threads_per_grid): 
            if data[i, 0] < threshold:
                data[i, 0] = val
    elif m == 1:
        for i in range(ix, n, threads_per_grid): 
            if data[0, i] < threshold:
                data[0, i] = val


                
@cuda.jit
def kernel_element_multiply_1D(A, B, C):
    '''
    A = (m, 1) or (1, n)
    B = (m, 1) or (1, n)
    C = (m, 1) or (1, n)
    '''
    ix = cuda.grid(1)
    threads_per_grid=cuda.gridsize(1)

    m, n = A.shape        
    if n == 1:
        for i in range(ix, m, threads_per_grid): 
            C[i, 0] = A[i, 0] * B[i, 0]
    elif m == 1:
        for i in range(ix, n, threads_per_grid): 
            C[0, i] = A[0, i] * B[0, i]
    
@cuda.jit
def kernel_element_divide_1D(A, B, C):
    '''
    A = (m, 1) or (1, n) 
    B = (m, 1) or (1, n)
    C = (m, 1) or (1, n)
    '''
    ix = cuda.grid(1)
    threads_per_grid=cuda.gridsize(1)

    m, n = A.shape    
    if n == 1:
        for i in range(ix, m, threads_per_grid): 
            if B[i, 0] == 0:
                C[i, 0] = 0
            else:
                C[i, 0] = A[i, 0] / B[i, 0]
    elif m == 1:
        for i in range(ix, n, threads_per_grid):
            if B[0, i] == 0:
                C[0, i] = 0
            else: 
                C[0, i] = A[0, i] / B[0, i]


@cuda.jit(device=True)
def cuda_retrieve_data_mask_cord(data_shape, mask_shape, sli, row, col):
    s0 = data_shape
    s = mask_shape
    xs = int(row)
    xe = int(row + s[1])
    ys = int(col - floor(s[2]/2))
    ye = int(col + floor(s[2]/2)+1)
    zs = int(sli - floor(s[0]/2))
    ze = int(sli + floor(s[0]/2)+1)
    ms = mask_shape
    m_xs = 0   
    m_xe = ms[1]

    m_ys = 0
    m_ye = ms[2]

    m_zs = 0
    m_ze = ms[0]
    if xs < 0:
        m_xs = -xs
        xs = 0
    if xe > s0[1]:
        m_xe = s0[1] - xe + ms[1]
        xe = s0[1]
    if ys < 0:
        m_ys = -ys
        ys = 0
    if ye > s0[2]:
        m_ye = s0[2] - ye + ms[2]
        ye = s0[2]
    if zs < 0:
        m_zs = -zs
        zs = 0
    if ze > s0[0]:
        m_ze = s0[0] - ze + ms[0]
        ze = s0[0]
    return zs, ze, xs, xe, ys, ye, m_zs, m_ze, m_xs, m_xe, m_ys, m_ye














#
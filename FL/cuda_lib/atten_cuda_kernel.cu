#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// ==========================================================
// Original attenuation kernel (UNCHANGED)
// ==========================================================

__global__ void atten_kernel(
    const float* __restrict__ mu,      // (K, Z, X, Y)
    const float* __restrict__ mask,    // (Mz, Mx, My)
    float* __restrict__ out,            // (K, Z, X, Y)
    int K, int Z, int X, int Y,
    int Mz, int Mx, int My
) {
    int k   = blockIdx.x;
    int sli = blockIdx.y;
    int row = blockIdx.z;
    int col = threadIdx.x;

    if (col >= Y) return;

    float acc = 0.0f;

    int hz = Mz / 2;
    int hx = Mx / 2;
    int hy = My / 2;

    for (int dz = 0; dz < Mz; dz++) {
        int z = sli + dz - hz;
        if (z < 0 || z >= Z) continue;

        for (int dx = 0; dx < Mx; dx++) {
            int x = row + dx;
            if (x < 0 || x >= X) continue;

            for (int dy = 0; dy < My; dy++) {
                int y = col + dy - hy;
                if (y < 0 || y >= Y) continue;

                int mu_idx =
                    ((k * Z + z) * X + x) * Y + y;

                int mask_idx =
                    (dz * Mx + dx) * My + dy;

                acc += mu[mu_idx] * mask[mask_idx];
            }
        }
    }

    int out_idx =
        ((k * Z + sli) * X + row) * Y + col;

    out[out_idx] = acc;
}


void atten_cuda_launcher(
    torch::Tensor mu,
    torch::Tensor mask,
    torch::Tensor out
) {
    int K = mu.size(0);
    int Z = mu.size(1);
    int X = mu.size(2);
    int Y = mu.size(3);

    int Mz = mask.size(0);
    int Mx = mask.size(1);
    int My = mask.size(2);

    dim3 blocks(K, Z, X);
    dim3 threads(Y);



    atten_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        mask.data_ptr<float>(),
        out.data_ptr<float>(),
        K, Z, X, Y,
        Mz, Mx, My
    );
}

// ==========================================
// forward_emission_kernel
// ==========================================
__global__ void forward_emission_kernel(
    const float* atten,   // (n_angle, H, W)
    const float* C,       // (n_ref, H, W)
    const float* em_cs,   // (n_angle, n_ref)
    const float* theta,   // (n_angle)
    float* Pf,            // (n_angle, W)
    int n_angle, int H, int W, int n_ref
) {
    int i = blockIdx.x;  // angle index
    int q = threadIdx.x + blockIdx.y * blockDim.x;
    if (q >= W) return;

    float cx = (W - 1) * 0.5f;
    float cy = (H - 1) * 0.5f;

    float th = theta[i];
    float c = cosf(-th);
    float s = sinf(-th);

    float acc = 0.0f;

    for (int row = 0; row < H; ++row) {
        float x = c * (row - cy) - s * (q - cx) + cy;
        float y = s * (row - cy) + c * (q - cx) + cx;

        // ===== SAFE CLAMP =====

        x = fminf(fmaxf(x, 0.f), H - 1.001f);
        y = fminf(fmaxf(y, 0.f), W - 1.001f);

        int x0 = (int)x;
        int y0 = (int)y;
        float dx = x - x0;
        float dy = y - y0;

        float val = 0.0f;
        for (int j = 0; j < n_ref; ++j) {
            const float* Cj = C + j * H * W;
            float interp =
                Cj[x0 * W + y0]       * (1 - dx) * (1 - dy) +
                Cj[x0 * W + y0 + 1]   * (1 - dx) * dy +
                Cj[(x0 + 1) * W + y0] * dx * (1 - dy) +
                Cj[(x0 + 1) * W + y0 + 1] * dx * dy;

            val += em_cs[i * n_ref + j] * interp;
        }

        acc += atten[i * H * W + row * W + q] * val;
    }

    Pf[i * W + q] = acc;
}


// ==========================================
// launcher
// ==========================================
void forward_emission_launcher(
    torch::Tensor atten,
    torch::Tensor C,
    torch::Tensor em_cs,
    torch::Tensor theta,
    torch::Tensor Pf
) {
    int n_angle = atten.size(0);
    int H = atten.size(1);
    int W = atten.size(2);
    int n_ref = C.size(0);

    const int threads = 256;
    dim3 blocks(n_angle, (W + threads - 1) / threads);

    forward_emission_kernel<<<blocks, threads>>>(
        atten.data_ptr<float>(),
        C.data_ptr<float>(),
        em_cs.data_ptr<float>(),
        theta.data_ptr<float>(),
        Pf.data_ptr<float>(),
        n_angle, H, W, n_ref
    );
}


// ==========================================
// backward_emission_kernel
// ==========================================
__global__ void backward_emission_kernel(
    const float* __restrict__ atten,   // (n_angle, H, W)
    const float* __restrict__ ratio,   // (n_angle, W)
    const float* __restrict__ em_cs,   // (n_angle, n_ref)
    const float* __restrict__ theta,   // (n_angle)
    float* __restrict__ out,           // (n_ref, H, W)
    int n_angle, int H, int W, int n_ref
) {
    int i = blockIdx.x;                             // angle
    int q = threadIdx.x + blockIdx.y * blockDim.x; // detector bin
    if (q >= W) return;

    float cx = (W - 1) * 0.5f;
    float cy = (H - 1) * 0.5f;

    float th = theta[i];
    float c = cosf(th);
    float s = sinf(th);

    float r = ratio[i * W + q];

    for (int row_det = 0; row_det < H; ++row_det) {

        // --- adjoint-consistent rotation ---
        float col_img = c * (q - cx) - s * (row_det - cy) + cx;
        float row_img = s * (q - cx) + c * (row_det - cy) + cy;

        // --- safe clamp ---
        col_img = fminf(fmaxf(col_img, 0.f), W - 1.001f);
        row_img = fminf(fmaxf(row_img, 0.f), H - 1.001f);

        int x0 = (int)col_img;
        int y0 = (int)row_img;
        float dx = col_img - x0;
        float dy = row_img - y0;

        float a = atten[i * H * W + row_det * W + q];

        for (int j = 0; j < n_ref; ++j) {
            float w = em_cs[i * n_ref + j] * a * r;

            atomicAdd(&out[j * H * W + y0 * W + x0],
                      w * (1 - dx) * (1 - dy));

            atomicAdd(&out[j * H * W + y0 * W + (x0 + 1)],
                      w * dx * (1 - dy));

            atomicAdd(&out[j * H * W + (y0 + 1) * W + x0],
                      w * (1 - dx) * dy);

            atomicAdd(&out[j * H * W + (y0 + 1) * W + (x0 + 1)],
                      w * dx * dy);
        }
    }
}





// ==========================================
// launcher
// ==========================================
void backward_emission_launcher(
    torch::Tensor atten,
    torch::Tensor ratio,
    torch::Tensor em_cs,
    torch::Tensor theta,
    torch::Tensor out
) {
    int n_angle = atten.size(0);
    int H = atten.size(1);
    int W = atten.size(2);

    const int threads = 256;
    dim3 blocks(n_angle, (W + threads - 1) / threads);

    cudaMemset(out.data_ptr<float>(), 0,
               out.numel() * sizeof(float));

    backward_emission_kernel<<<blocks, threads>>>(
        atten.data_ptr<float>(),
        ratio.data_ptr<float>(),
        em_cs.data_ptr<float>(),
        theta.data_ptr<float>(),
        out.data_ptr<float>(),
        n_angle, H, W, out.size(0)
    );
}


// ==========================================
// Batch forward emission kernel
// ==========================================
__device__ inline float bilinear_sample(
    const float* img,
    int H, int W,
    float y, float x
) {
    x = fminf(fmaxf(x, 0.f), W - 1.001f);
    y = fminf(fmaxf(y, 0.f), H - 1.001f);

    int x0 = floorf(x);
    int y0 = floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[y0 * W + x0];
    float v01 = (x1 < W) ? img[y0 * W + x1] : v00;
    float v10 = (y1 < H) ? img[y1 * W + x0] : v00;
    float v11 = (x1 < W && y1 < H) ? img[y1 * W + x1] : v00;

    return (1 - dx) * (1 - dy) * v00 +
           dx * (1 - dy) * v01 +
           (1 - dx) * dy * v10 +
           dx * dy * v11;
}

__global__ void forward_emission_batch_kernel(
    const float* __restrict__ atten,   // (A, B, H, W)
    const float* __restrict__ C,       // (R, B, H, W)
    const float* __restrict__ em_cs,   // (A, R)
    const float* __restrict__ theta,   // (A)
    float* __restrict__ Pf,            // (A, B, W)
    int A, int B, int R, int H, int W
) {
    int w = blockIdx.x * blockDim.x + threadIdx.x;  // detector bin
    int h = blockIdx.y * blockDim.y + threadIdx.y;  // row in image
    int b = blockIdx.z;                             // slice index
    if (w >= W || h >= H || b >= B) return;

    // Loop over angles
    for (int a = 0; a < A; ++a) {
        float th = theta[a];
        float c = cosf(-th);
        float s = sinf(-th);

        float cx = (W - 1) * 0.5f;
        float cy = (H - 1) * 0.5f;

        float x_rot = c * (h - cy) - s * (w - cx) + cy;
        float y_rot = s * (h - cy) + c * (w - cx) + cx;

        float sum_val = 0.0f;

        for (int r = 0; r < R; ++r) {
            int idx_C = ((r * B + b) * H * W);
            float val = bilinear_sample(C + idx_C, H, W, x_rot, y_rot);
            float cs = em_cs[a * R + r];

            int idx_att = ((a * B + b) * H + h) * W + w;
            sum_val += cs * val * atten[idx_att];
        }

        int idx_Pf = (a * B + b) * W + w;
        atomicAdd(&Pf[idx_Pf], sum_val);
    }
}



// ==========================================
// batch forward emission launcher
// ==========================================
void forward_emission_batch_launcher(
    torch::Tensor atten,  // (A, B, H, W)
    torch::Tensor C,      // (R, B, H, W)
    torch::Tensor em_cs,  // (A, R)
    torch::Tensor theta,  // (A)
    torch::Tensor Pf      // (A, B, W)
) {
    const int A = atten.size(0); // n_angle
    const int B = atten.size(1); // batch slices
    const int H = atten.size(2); // image rows
    const int W = atten.size(3); // image cols
    const int R = C.size(0);     // n_ref

    // Define 2D thread block for H x W
    const int THREAD_X = 16; // W direction
    const int THREAD_Y = 16; // H direction

    dim3 threads(THREAD_X, THREAD_Y);
    dim3 blocks(
        (W + THREAD_X - 1) / THREAD_X,
        (H + THREAD_Y - 1) / THREAD_Y,
        B  // batch dimension
    );

    // Zero output first
    cudaMemset(Pf.data_ptr<float>(), 0, Pf.numel() * sizeof(float));

    // Launch kernel
    forward_emission_batch_kernel<<<blocks, threads>>>(
        atten.data_ptr<float>(),
        C.data_ptr<float>(),
        em_cs.data_ptr<float>(),
        theta.data_ptr<float>(),
        Pf.data_ptr<float>(),
        A, B, R, H, W
    );

    // Optional: check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA forward_emission_batch_kernel error: %s\n", cudaGetErrorString(err));
};

// ==========================================
// batch backward emission 
// ==========================================
__global__ void backward_emission_batch_kernel(
    const float* __restrict__ atten,   // (A, B, H, W)
    const float* __restrict__ ratio,   // (A, B, W)
    const float* __restrict__ em_cs,   // (A, R)
    const float* __restrict__ theta,   // (A)
    float* __restrict__ out,           // (R, B, H, W)
    int A, int B, int R, int H, int W
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x; // detector bin
    int b = blockIdx.y;                            // batch slice
    int a = blockIdx.z;                            // angle

    if (q >= W || b >= B || a >= A) return;

    float cx = (W - 1) * 0.5f;
    float cy = (H - 1) * 0.5f;

    float th = theta[a];
    float c = cosf(th);
    float s = sinf(th);

    float r = ratio[(a * B + b) * W + q];

    for (int row_det = 0; row_det < H; ++row_det) {

        // ---- adjoint-consistent inverse rotation ----
        float col_img = c * (q - cx) - s * (row_det - cy) + cx;
        float row_img = s * (q - cx) + c * (row_det - cy) + cy;

        // ---- clamp safely ----
        col_img = fminf(fmaxf(col_img, 0.f), W - 1.001f);
        row_img = fminf(fmaxf(row_img, 0.f), H - 1.001f);

        int x0 = (int)col_img;
        int y0 = (int)row_img;
        float dx = col_img - x0;
        float dy = row_img - y0;

        float a_val =
            atten[((a * B + b) * H + row_det) * W + q];

        for (int j = 0; j < R; ++j) {
            float w = em_cs[a * R + j] * a_val * r;

            int base = ((j * B + b) * H + y0) * W + x0;

            atomicAdd(&out[base],
                      w * (1 - dx) * (1 - dy));
            atomicAdd(&out[base + 1],
                      w * dx * (1 - dy));
            atomicAdd(&out[base + W],
                      w * (1 - dx) * dy);
            atomicAdd(&out[base + W + 1],
                      w * dx * dy);
        }
    }
}



// ==========================================
// batch backward emission launcher
// ==========================================
void backward_emission_batch_launcher(
    torch::Tensor atten,
    torch::Tensor ratio,
    torch::Tensor em_cs,
    torch::Tensor theta,
    torch::Tensor out
) {
    const int A = atten.size(0);
    const int B = atten.size(1);
    const int H = atten.size(2);
    const int W = atten.size(3);
    const int R = out.size(0);

    cudaMemset(
        out.data_ptr<float>(),
        0,
        out.numel() * sizeof(float)
    );

    const int THREADS = 256;
    dim3 blocks(
        (W + THREADS - 1) / THREADS,
        B,
        A
    );

    backward_emission_batch_kernel<<<blocks, THREADS>>>(
        atten.data_ptr<float>(),
        ratio.data_ptr<float>(),
        em_cs.data_ptr<float>(),
        theta.data_ptr<float>(),
        out.data_ptr<float>(),
        A, B, R, H, W
    );
}



// ==========================================
// Huber gradient kernel
// ==========================================

__global__ void huber_grad_kernel(
    const float* __restrict__ C,    // (n_ref, H, W)
    float* __restrict__ grad,        // (n_ref, H, W)
    int H, int W, int n_ref,
    float delta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    if (idx >= HW * n_ref) return;

    int j = idx / HW;          // reference index
    int rem = idx % HW;
    int r = rem / W;
    int c = rem % W;

    int base = j * HW + r * W + c;
    float Cp = C[base];

    float g = 0.0f;

    // ---- neighbors ----
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    for (int k = 0; k < 4; ++k) {
        int rr = r + dr[k];
        int cc = c + dc[k];
        if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;

        float diff = Cp - C[j * HW + rr * W + cc];

        if (fabsf(diff) <= delta)
            g += diff;
        else
            g += delta * copysignf(1.0f, diff);
    }

    grad[base] = g;

}

// ==========================================
// launcher
// ==========================================

void huber_grad_launcher(
    torch::Tensor C,
    torch::Tensor grad,
    float delta
) {
    int n_ref = C.size(0);
    int H = C.size(1);
    int W = C.size(2);

    int total = n_ref * H * W;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    huber_grad_kernel<<<blocks, threads>>>(
        C.data_ptr<float>(),
        grad.data_ptr<float>(),
        H, W, n_ref,
        delta
    );
}

// ==========================================
// Batch Huber gradient kernel
// ==========================================
__global__ void huber_grad_batch_kernel(
    const float* __restrict__ C,    // (n_ref, B, H, W)
    float* __restrict__ grad,        // (n_ref, B, H, W)
    int B, int H, int W, int n_ref,
    float delta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int HW = H * W;
    int BHW = B * HW;
    int total = n_ref * BHW;

    if (idx >= total) return;

    // ------------------------------------
    // unravel index
    // ------------------------------------
    int j   = idx / BHW;             // ref index
    int rem = idx % BHW;

    int b   = rem / HW;              // batch index
    rem     = rem % HW;

    int r   = rem / W;
    int c   = rem % W;

    int base = ((j * B + b) * H + r) * W + c;
    float Cp = C[base];

    float g = 0.0f;

    // ---- spatial neighbors only ----
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = {0, 0, -1, 1};

    for (int k = 0; k < 4; ++k) {
        int rr = r + dr[k];
        int cc = c + dc[k];

        if (rr < 0 || rr >= H || cc < 0 || cc >= W)
            continue;

        int nb = ((j * B + b) * H + rr) * W + cc;
        float diff = Cp - C[nb];

        if (fabsf(diff) <= delta)
            g += diff;
        else
            g += delta * copysignf(1.0f, diff);
    }

    grad[base] = g;
}


// ==========================================
// Batch Huber gradient kernel launcher
// ==========================================
void huber_grad_batch_launcher(
    torch::Tensor C,     // (n_ref, B, H, W)
    torch::Tensor grad,  // (n_ref, B, H, W)
    float delta
) {
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(grad.is_cuda(), "grad must be CUDA");
    TORCH_CHECK(C.dim() == 4, "C must be (n_ref, B, H, W)");

    int n_ref = C.size(0);
    int B     = C.size(1);
    int H     = C.size(2);
    int W     = C.size(3);

    int total = n_ref * B * H * W;

    const int threads = 256;
    const int blocks  = (total + threads - 1) / threads;

    huber_grad_batch_kernel<<<blocks, threads>>>(
        C.data_ptr<float>(),
        grad.data_ptr<float>(),
        B, H, W, n_ref,
        delta
    );
}

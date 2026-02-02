#include <torch/extension.h>

// ================= Existing attenuation =================

void atten_cuda_launcher(
    torch::Tensor mu,
    torch::Tensor mask,
    torch::Tensor out
);


torch::Tensor cal_atten_cuda(
    torch::Tensor mu,
    torch::Tensor mask
) {
    TORCH_CHECK(mu.is_cuda(), "mu must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu must be float32");
    TORCH_CHECK(mask.dtype() == torch::kFloat32, "mask must be float32");

    // Allocate output tensor
    auto out = torch::zeros_like(mu);

    // Launch CUDA kernel
    atten_cuda_launcher(mu, mask, out);

    return out;
}


// ================= New emission projectors =================
// ---------------------------------------------------------
// CUDA launcher declarations
// ---------------------------------------------------------
void forward_emission_launcher(
    torch::Tensor atten,
    torch::Tensor C,
    torch::Tensor em_cs,
    torch::Tensor theta,
    torch::Tensor Pf
);

void backward_emission_launcher(
    torch::Tensor atten,
    torch::Tensor ratio,
    torch::Tensor em_cs,
    torch::Tensor theta,
    torch::Tensor out
);

void huber_grad_launcher(
    torch::Tensor C,
    torch::Tensor grad,
    float delta
);

void huber_grad_batch_launcher(
    torch::Tensor C,
    torch::Tensor grad,
    float delta
);

void forward_emission_batch_launcher(
    torch::Tensor atten,   // (n_angle, H, W)
    torch::Tensor C,       // (n_ref, n_sli, H, W)
    torch::Tensor em_cs,   // (n_angle, n_ref)
    torch::Tensor theta,   // (n_angle)
    torch::Tensor Pf       // (n_angle, n_sli, W)
);

void backward_emission_batch_launcher(
    torch::Tensor atten,
    torch::Tensor ratio,
    torch::Tensor em_cs,
    torch::Tensor theta,
    torch::Tensor out
);

// =========================================================
// Forward emission (Python API)
// =========================================================
torch::Tensor forward_emission(
    torch::Tensor atten,   // (n_angle, H, W)
    torch::Tensor C,       // (n_ref, H, W)
    torch::Tensor em_cs,   // (n_angle, n_ref)
    torch::Tensor theta    // (n_angle)
) {
    TORCH_CHECK(atten.is_cuda(), "atten must be CUDA");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(em_cs.is_cuda(), "em_cs must be CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta must be CUDA");

    TORCH_CHECK(atten.dim() == 3, "atten must be (n_angle, H, W)");
    TORCH_CHECK(C.dim() == 3, "C must be (n_ref, H, W)");
    TORCH_CHECK(em_cs.dim() == 2, "em_cs must be (n_angle, n_ref)");
    TORCH_CHECK(theta.dim() == 1, "theta must be (n_angle)");

    auto atten_c = atten.contiguous();
    auto C_c     = C.contiguous();

    int64_t n_angle = atten_c.size(0);
    int64_t W       = atten_c.size(2);

    auto Pf = torch::zeros({n_angle, W}, atten_c.options());

    forward_emission_launcher(atten_c, C_c, em_cs, theta, Pf);
    return Pf;
}

// =========================================================
// Backward emission (Python API)
// =========================================================
torch::Tensor backward_emission(
    torch::Tensor atten,   // (n_angle, H, W)
    torch::Tensor ratio,   // (n_angle, W)
    torch::Tensor em_cs,   // (n_angle, n_ref)
    torch::Tensor theta    // (n_angle)
) {
    TORCH_CHECK(atten.is_cuda(), "atten must be CUDA");
    TORCH_CHECK(ratio.is_cuda(), "ratio must be CUDA");
    TORCH_CHECK(em_cs.is_cuda(), "em_cs must be CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta must be CUDA");

    int64_t n_ref = em_cs.size(1);
    int64_t H = atten.size(1);
    int64_t W = atten.size(2);

    auto out = torch::zeros(
        {n_ref, H, W},
        atten.options()
    );

    backward_emission_launcher(atten, ratio, em_cs, theta, out);
    return out;
}


// =========================================================
// Batch Forward emission (Python API)
// =========================================================

torch::Tensor forward_emission_batch(
    torch::Tensor atten,   // (n_angle, B, H, W)
    torch::Tensor C,       // (n_ref,   B, H, W)
    torch::Tensor em_cs,   // (n_angle, n_ref)
    torch::Tensor theta    // (n_angle)
) {
    TORCH_CHECK(atten.is_cuda(), "atten must be CUDA");
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(em_cs.is_cuda(), "em_cs must be CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta must be CUDA");

    TORCH_CHECK(atten.dim() == 4, "atten must be (n_angle, B, H, W)");
    TORCH_CHECK(C.dim() == 4, "C must be (n_ref, B, H, W)");
    TORCH_CHECK(em_cs.dim() == 2, "em_cs must be (n_angle, n_ref)");
    TORCH_CHECK(theta.dim() == 1, "theta must be (n_angle)");

    auto atten_c = atten.contiguous();
    auto C_c     = C.contiguous();

    int64_t n_angle = atten_c.size(0);
    int64_t B       = atten_c.size(1);
    int64_t H       = atten_c.size(2);
    int64_t W       = atten_c.size(3);

    TORCH_CHECK(C_c.size(1) == B, "batch size mismatch");
    TORCH_CHECK(C_c.size(2) == H, "H mismatch");
    TORCH_CHECK(C_c.size(3) == W, "W mismatch");
    TORCH_CHECK(em_cs.size(0) == n_angle, "n_angle mismatch");
    TORCH_CHECK(em_cs.size(1) == C_c.size(0), "n_ref mismatch");

    auto Pf = torch::zeros(
        {n_angle, B, W},
        atten_c.options()
    );

    forward_emission_batch_launcher(
        atten_c, C_c, em_cs, theta, Pf
    );

    return Pf;
}

// =========================================================
// Batch backward emission (Python API)
// =========================================================

torch::Tensor backward_emission_batch(
    torch::Tensor atten,   // (A, B, H, W)
    torch::Tensor ratio,   // (A, B, W)
    torch::Tensor em_cs,   // (A, R)
    torch::Tensor theta    // (A)
) {
    TORCH_CHECK(atten.is_cuda(), "atten must be CUDA");
    TORCH_CHECK(ratio.is_cuda(), "ratio must be CUDA");
    TORCH_CHECK(em_cs.is_cuda(), "em_cs must be CUDA");
    TORCH_CHECK(theta.is_cuda(), "theta must be CUDA");

    int64_t R = em_cs.size(1);
    int64_t B = atten.size(1);
    int64_t H = atten.size(2);
    int64_t W = atten.size(3);

    auto out = torch::zeros(
        {R, B, H, W},
        atten.options()
    );

    backward_emission_batch_launcher(
        atten, ratio, em_cs, theta, out
    );

    return out;
}



void huber_grad(
    torch::Tensor C,
    torch::Tensor grad,
    float delta
) {
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(grad.is_cuda(), "grad must be CUDA");

    huber_grad_launcher(C, grad, delta);
}

void huber_grad_batch(
    torch::Tensor C,
    torch::Tensor grad,
    float delta
) {
    TORCH_CHECK(C.is_cuda(), "C must be CUDA");
    TORCH_CHECK(grad.is_cuda(), "grad must be CUDA");

    huber_grad_batch_launcher(C, grad, delta);
}


// =========================================================
// MLEM CUDA with Huber regularization
// =========================================================
torch::Tensor mlem_cuda(
    torch::Tensor C_init,
    torch::Tensor atten,
    torch::Tensor em_cs,
    torch::Tensor theta,
    torch::Tensor I,
    int64_t n_iter,
    double beta = 1e-3,
    double delta = 0.01
){
    auto C = C_init.clone();
    auto ones = torch::ones_like(I);

    // sensitivity = Hᵀ 1
    auto sens = backward_emission(atten, ones, em_cs, theta);
    sens.clamp_min_(1e-6);

    auto grad = torch::zeros_like(C);

    for (int iter = 0; iter < n_iter; ++iter) {
        auto Pf = forward_emission(atten, C, em_cs, theta);
        Pf.clamp_min_(1e-6);

        auto ratio = I / Pf;
        auto back = backward_emission(atten, ratio, em_cs, theta);

        if (beta > 0.0) {
            huber_grad(C, grad, delta);
            C.mul_(back).div_(sens + beta * grad);
        } else {
            C.mul_(back).div_(sens);
        }

        C.clamp_min_(0.0);
    }
    return C;

}

torch::Tensor mlem_cuda_batch(
    torch::Tensor C_init,   // (n_ref, B, H, W)
    torch::Tensor atten,    // (n_angle, B, H, W)
    torch::Tensor em_cs,    // (n_angle, n_ref)
    torch::Tensor theta,    // (n_angle)
    torch::Tensor I,        // (n_angle, B, W)
    int64_t n_iter,
    double beta  = 1e-3,
    double delta = 0.01
) {
    TORCH_CHECK(C_init.is_cuda(), "C_init must be CUDA");
    TORCH_CHECK(atten.is_cuda(),  "atten must be CUDA");
    TORCH_CHECK(em_cs.is_cuda(),  "em_cs must be CUDA");
    TORCH_CHECK(theta.is_cuda(),  "theta must be CUDA");
    TORCH_CHECK(I.is_cuda(),      "I must be CUDA");

    TORCH_CHECK(C_init.dim() == 4, "C_init must be (n_ref, B, H, W)");
    TORCH_CHECK(atten.dim()  == 4, "atten must be (n_angle, B, H, W)");
    TORCH_CHECK(I.dim()      == 3, "I must be (n_angle, B, W)");

    auto C = C_init.clone();

    // ----------------------------
    // Sensitivity = Hᵀ 1
    // ----------------------------
    auto ones = torch::ones_like(I);

    auto sens = backward_emission_batch(
        atten, ones, em_cs, theta
    );

    sens.clamp_min_(1e-6);

    auto grad = torch::zeros_like(C);

    // ----------------------------
    // MLEM iterations
    // ----------------------------
    for (int iter = 0; iter < n_iter; ++iter) {

        auto Pf = forward_emission_batch(
            atten, C, em_cs, theta
        );
        Pf.clamp_min_(1e-6);

        auto ratio = I / Pf;

        auto back = backward_emission_batch(
            atten, ratio, em_cs, theta
        );

        if (beta > 0.0) {
            huber_grad_batch(C, grad, delta);
            C.mul_(back).div_(sens + beta * grad);
        } else {
            C.mul_(back).div_(sens);
        }

        C.clamp_min_(0.0);
    }

    return C;
}

// ==========================================================
// Pybind
// ==========================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cal_atten_cuda", &cal_atten_cuda, "Calculate attenuation CUDA");
    m.def("forward_emission", &forward_emission, "Forward emission (CUDA)");
    m.def("backward_emission", &backward_emission, "Backward emission (CUDA)");
    //m.def("mlem_cuda", &mlem_cuda, "MLEM reconstruction (CUDA)");
    m.def(
        "mlem_cuda",
        &mlem_cuda,
        py::arg("C_init"),
        py::arg("atten"),
        py::arg("em_cs"),
        py::arg("theta"),
        py::arg("I"),
        py::arg("n_iter"),
        py::arg("beta") = 1e-3,
        py::arg("delta") = 0.01,
        "MLEM reconstruction with optional Huber regularization"
    );
    m.def(
    "forward_emission_batch",
    &forward_emission_batch,
    "Forward emission with batch slices (CUDA)"
    );
    m.def(
    "backward_emission_batch",
    &backward_emission_batch,
    "Backward emission with batch slices (CUDA)"
    );
    m.def(
    "mlem_cuda_batch",
    &mlem_cuda_batch,
    py::arg("C_init"),
    py::arg("atten"),
    py::arg("em_cs"),
    py::arg("theta"),
    py::arg("I"),
    py::arg("n_iter"),
    py::arg("beta")  = 1e-3,
    py::arg("delta") = 0.01,
    "Batch MLEM reconstruction with optional Huber regularization (CUDA)"
    );

}


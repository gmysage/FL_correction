
import torch.utils.checkpoint as cp
import torch

'''
1. Poisson loss:
if AX = I, Poisson loss is defined as :
    L = (AX - I*log(AX))
it is stable than MSE loss of L=(AX - I)**2   

2. 
Jacobian / Lipschitz regularization (advanced but powerful)
∥∇C​CNN(C)∥≤1
e.g.,
eps = torch.randn_like(C)
loss_jac = ((denoiser(C + 1e-3 * eps) - denoiser(C)).norm()
           / eps.norm())

3. Denoiser consistency loss (very important)
Ldenoise​=∥CNN(C)−C∥
loss_consistency = ((C_denoised - C).pow(2)).mean()

'''
class MLEMLayer(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, C, atten, em_cs, theta, I, sens):
        Pf = FL.forward_emission_autograd(atten, C, em_cs, theta)
        Pf = Pf.clamp_min(1e-6)

        ratio = I / Pf
        back = FL.backward_emission_autograd(atten, ratio, em_cs, theta)

        C_mlem = C * back / sens
        C_mlem = C_mlem.clamp_min(0.0)

        # CNN regularization
        C_next = self.denoiser(C_mlem)

        return C_next.clamp_min(0.0)




class DenoiserCNN(torch.nn.Module):
    def __init__(self, n_ref, n_feat=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(n_ref, n_feat, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_feat, n_feat, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_feat, n_ref, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)   # residual learning

class UnrolledMLEM(torch.nn.Module):
    def __init__(self, n_iter, n_ref):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            MLEMLayer(DenoiserCNN(n_ref))
            for _ in range(n_iter)
        ])

    def forward(self, C0, atten, em_cs, theta, I):
        ones = torch.ones_like(I)
        sens = FL.backward_emission_autograd(atten, ones, em_cs, theta)

        C = C0
        for layer in self.layers:
            C = layer(C, atten, em_cs, theta, I, sens)
        return C


def poisson_loss(Pf, I):
    return (Pf - I * torch.log(Pf + 1e-6)).mean()


def denoiser_consistency_loss(C, C_d):
    return ((C_d - C) ** 2).mean()




def jacobian_regularization(denoiser, C, eps=1e-3):
    noise = torch.randn_like(C)
    C_perturbed = C + eps * noise

    D1 = denoiser(C)
    D2 = denoiser(C_perturbed)

    num = (D2 - D1).norm()
    den = (eps * noise).norm() + 1e-6
    return num / den

####


# Sensitivity precomputation (once)
with torch.no_grad():
    ones = torch.ones_like(I)
    sens = atten_cuda.backward_emission(atten, ones, em_cs, theta)
    sens.clamp_min_(1e-6)

WARMUP_ITERS = 10        # CNN frozen
PARTIAL_TUNE_ITERS = 10 # unfreeze last layer
TOTAL_ITERS = 40

### freeze denoiser
for p in denoiser.parameters():
    p.requires_grad = False

### unfreeze last layer
def unfreeze_last_layer(model):
    for name, p in model.named_parameters():
        if "last" in name or "out" in name:
            p.requires_grad = True


optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, denoiser.parameters()),
    lr=1e-5
)           
###
C = C_init.clone().cuda()

for k in range(TOTAL_ITERS):

    # ===============================
    # 1. MLEM physics update
    # ===============================
    with torch.no_grad():
        C = mlem_step(C, atten, em_cs, theta, I, sens)

    # ===============================
    # 2. CNN denoising
    # ===============================
    C.requires_grad_(k >= WARMUP_ITERS) # this does not make C a parameter — it just allows gradients to flow through C into the CNN weights.
 
    C_cnn = C_d.unsqueeze(0)  # (1, n_ref, H, W)
    C_d_cnn = denoiser(C_cnn) # (1, n_ref, H, W)
    C_d = C_d_cnn.squeeze(0)  # (n_ref, H, W)

    # ===============================
    # 3. Optional CNN fine-tuning
    # ===============================
    if k >= WARMUP_ITERS:

        if k == WARMUP_ITERS:
            unfreeze_last_layer(denoiser)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, denoiser.parameters()),
                lr=1e-5
            )

        Pf = atten_cuda.forward_emission(atten, C_d, em_cs, theta)
        Pf.clamp_min_(1e-6)

        loss_data = poisson_loss(Pf, I)
        loss_cons = denoiser_consistency_loss(C, C_d)
        loss_jac = jacobian_regularization(denoiser, C)

        loss = (
            loss_data
            + 0.1 * loss_cons
            + 1e-3 * loss_jac
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ===============================
    # 4. Detach for next iteration
    # ===============================
    C = C_d.detach()

    print(
        f"Iter {k:02d} | "
        f"Data {loss_data.item():.4e} | "
        f"Cons {loss_cons.item():.4e} | "
        f"Jac {loss_jac.item():.4e}"
    )





######################################################################
# if including updating atten_xary in the loop
######################################################################
C = C_init.clone().cuda()

denoiser.eval()   # start frozen
optimizer_cnn = None

for k in range(TOTAL_ITERS):

    # =====================================================
    # 1. Physics update (NO autograd)
    # =====================================================
    with torch.no_grad():

        atten_xray = cal_xray_atten(C)
        atten = atten_xray * atten_xrf

        C = mlem_step(C, atten, em_cs, theta, I, sens)
        C.clamp_min_(0.0)

    # =====================================================
    # 2. CNN denoising (forward only)
    # =====================================================
    C_d = denoiser(C)

    # =====================================================
    # 3. Optional CNN fine-tuning
    # =====================================================
    if k >= WARMUP_ITERS:

        if k == WARMUP_ITERS:
            denoiser.train()
            unfreeze_last_layer(denoiser)

            optimizer_cnn = torch.optim.Adam(
                filter(lambda p: p.requires_grad, denoiser.parameters()),
                lr=1e-5
            )

        # ----- physics loss uses UPDATED attenuation -----
        atten_xray = cal_xray_atten(C_d.detach())
        atten = atten_xray * atten_xrf

        Pf = atten_cuda.forward_emission(
            atten, C_d, em_cs, theta
        ).clamp_min(1e-6)

        loss_data = poisson_loss(Pf, I)
        loss_cons = denoiser_consistency_loss(C, C_d)
        loss_jac = jacobian_regularization(denoiser, C)

        loss = loss_data + 0.1 * loss_cons + 1e-3 * loss_jac

        optimizer_cnn.zero_grad()
        loss.backward()
        optimizer_cnn.step()

    # =====================================================
    # 4. Prepare for next iteration
    # =====================================================
    C = C_d.detach()
    
##################################################################
################ unrolled MELM with CNN as denoiser ##############
##################################################################


'''
1. Poisson loss:
if AX = I, Poisson loss is defined as :
    L = (AX - I*log(AX))
it is stable than MSE loss of L=(AX - I)**2   

2. 
Jacobian / Lipschitz regularization (advanced but powerful)
∥∇C​CNN(C)∥≤1
e.g.,
eps = torch.randn_like(C)
loss_jac = ((denoiser(C + 1e-3 * eps) - denoiser(C)).norm()
           / eps.norm())

3. Denoiser consistency loss (very important)
Ldenoise​=∥CNN(C)−C∥
loss_consistency = ((C_denoised - C).pow(2)).mean()

'''
class MLEMLayer(torch.nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    def forward(self, C, atten, em_cs, theta, I, sens):
        Pf = FL.forward_emission_autograd(atten, C, em_cs, theta)
        Pf = Pf.clamp_min(1e-6)

        ratio = I / Pf
        back = FL.backward_emission_autograd(atten, ratio, em_cs, theta)

        C_mlem = C * back / sens
        C_mlem = C_mlem.clamp_min(0.0)

        # 🔥 CNN regularization
        C_next = self.denoiser(C_mlem)

        return C_next.clamp_min(0.0)




class DenoiserCNN(torch.nn.Module):
    def __init__(self, n_ref, n_feat=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(n_ref, n_feat, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_feat, n_feat, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(n_feat, n_ref, 3, padding=1)
        )

    def forward(self, x):
        return x + self.net(x)   # residual learning

class UnrolledMLEM(torch.nn.Module):
    def __init__(self, n_iter, n_ref):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            MLEMLayer(DenoiserCNN(n_ref))
            for _ in range(n_iter)
        ])

    def forward(self, C0, atten, em_cs, theta, I):
        ones = torch.ones_like(I)
        sens = FL.backward_emission_autograd(atten, ones, em_cs, theta)

        C = C0
        for layer in self.layers:
            C = layer(C, atten, em_cs, theta, I, sens)
        return C


def poisson_loss(Pf, I):
    return (Pf - I * torch.log(Pf + 1e-6)).mean()


def denoiser_consistency_loss(C, C_d):
    return ((C_d - C) ** 2).mean()




def jacobian_regularization(denoiser, C, eps=1e-3):
    noise = torch.randn_like(C)
    C_perturbed = C + eps * noise

    D1 = denoiser(C)
    D2 = denoiser(C_perturbed)

    num = (D2 - D1).norm()
    den = (eps * noise).norm() + 1e-6
    return num / den

####



################################################
### sliding window 3D CNN denoiser
################################################
def robust_normalize_C(C, eps=1e-6, p=0.98):
    """
    C: (n_ref, B, H, W)
    """
    # Compute percentile per (ref, batch)
    flat = C.flatten(-2)
    scale_p = torch.quantile(flat, p, dim=-1, keepdim=True)
    scale_p = scale_p.unsqueeze(-1)

    # Fallback to mean if percentile is too small
    scale_m = C.mean(dim=(-2, -1), keepdim=True)

    scale = torch.where(
        scale_p > eps,
        scale_p,
        scale_m
    )

    return C / (scale + eps), scale

class RefAdaptivePercentile(nn.Module):
    def __init__(self, n_ref, p_init=0.95):
        super().__init__()
        self.p_raw = nn.Parameter(
            torch.full((n_ref,), p_init)
        )

    def forward(self):
        return torch.clamp(self.p_raw, 0.7, 0.99)


def adaptive_normalize_C_ref(C, p_module, eps=1e-6):
    p = p_module()                            # (n_ref,)
    flat = C.flatten(-2)

    scales = []
    for r in range(C.shape[0]):
        s = torch.quantile(flat[r], p[r], dim=-1, keepdim=True)
        scales.append(s.unsqueeze(-1))

    scale = torch.stack(scales, dim=0)
    scale = torch.clamp(scale, min=eps)

    return C / scale, scale


import torch
import torch.nn.functional as F

def sliding_window_ref_aware(
    C,                     # (n_ref, B, H, W)
    denoiser,
    p_module,              # RefAdaptivePercentile
    win_size=5,
    eps=1e-6
):
    """
    Reference-aware sliding-window 3D CNN denoiser
    with adaptive percentile normalization.

    Args:
        C:         (n_ref, B, H, W)
        denoiser:  RefAware3DCNN
        p_module:  RefAdaptivePercentile
        win_size:  sliding window size along B
    Returns:
        C_denoised: (n_ref, B, H, W)
    """
    n_ref, B, H, W = C.shape
    pad = win_size // 2

    # ------------------------------------------------
    # 1. Adaptive percentile normalization (NO inplace)
    # ------------------------------------------------
    Cn, scale = adaptive_normalize_C_ref(C, p_module, eps=eps)
    # Cn, scale: both differentiable

    # ------------------------------------------------
    # 2. Pad along B dimension
    # ------------------------------------------------
    C_pad = F.pad(
        Cn,
        (0, 0, 0, 0, pad, pad),
        mode="replicate"
    )  # (n_ref, B+2p, H, W)

    # ------------------------------------------------
    # 3. Sliding-window denoising
    # ------------------------------------------------
    out_slices = []

    for b in range(B):
        win = C_pad[:, b:b + win_size]      # (n_ref, win, H, W)
        win = win.unsqueeze(0)              # (1, n_ref, win, H, W)

        out = denoiser(win).squeeze(0)      # (n_ref, win, H, W)
        out_slices.append(out[:, pad])      # center slice

    C_out = torch.stack(out_slices, dim=1)  # (n_ref, B, H, W)

    # ------------------------------------------------
    # 4. De-normalize (NO inplace)
    # ------------------------------------------------
    return C_out * scale





class RefAware3DCNN(nn.Module):
    def __init__(self, n_ref, emb_dim=8, hidden=32):
        super().__init__()

        self.n_ref = n_ref
        self.emb = nn.Embedding(n_ref, emb_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        in_ch = 1 + emb_dim

        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv3d(hidden, 1, 3, padding=1)
        )

    def forward(self, x, ref_idx):
        """
        x:       (B, 1, D, H, W)
        ref_idx: scalar or (B,)
        """
        B, _, D, H, W = x.shape

        emb = self.emb(ref_idx)           # (B, E)
        emb = emb.view(B, -1, 1, 1, 1)
        emb = emb.expand(-1, -1, D, H, W)

        x = torch.cat([x, emb], dim=1)
        return self.alpha * self.net(x) + x


import torch
import torch.nn as nn
from torch.autograd import Function
import torch.utils.checkpoint as cp
# ------------------------------
# Unrolled batched MLEM
# ------------------------------


def checkpointed_denoiser(
    C,
    denoiser,
    p_module,
    win_size
):
    """
    Gradient-checkpointed reference-aware sliding-window denoiser
    with adaptive percentile normalization.
    """
    def fn(x):
        return sliding_window_ref_aware(
            x,
            denoiser=denoiser,
            p_module=p_module,
            win_size=win_size
        )

    return cp.checkpoint(fn, C)


'''

class UnrolledBatchMLEM(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        p_module: nn.Module,          # ⭐ NEW
        n_iter: int = 10,
        win_size: int = 5,
        beta: float = 0.1,
        delta: float = 0.01,
        warmup_iters: int = 3,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.denoiser = denoiser
        self.p_module = p_module      # ⭐ registered
        self.n_iter = n_iter
        self.win_size = win_size
        self.beta = beta
        self.delta = delta
        self.warmup_iters = warmup_iters
        self.use_checkpoint = use_checkpoint

    def forward(
        self,
        C_init,    # (n_ref, B, H, W)
        atten,     # (n_angle, B, H, W)
        em_cs,     # (n_angle, n_ref)
        theta,     # (n_angle,)
        I,         # (n_angle, B, W)
        fine_tune: bool = True
    ):
        """
        Fully unrolled batch MLEM with:
        - differentiable physics
        - adaptive percentile normalization
        - RefAware 3D CNN denoising
        """

        # ---- safe starting point (no inplace) ----
        C = C_init.clone()

        for k in range(self.n_iter):

            # --------------------------------------------------
            # Physics update (1 MLEM step, autograd-enabled)
            # --------------------------------------------------
            C = mlem_cuda_batch(
                C,
                atten,
                em_cs,
                theta,
                I,
                1,
                beta=self.beta,
                delta=self.delta
            )

            # --------------------------------------------------
            # CNN denoising (after warmup)
            # --------------------------------------------------
            if k >= self.warmup_iters:
                if self.use_checkpoint and fine_tune:
                    C = checkpointed_denoiser(
                        C,
                        denoiser=self.denoiser,
                        p_module=self.p_module,
                        win_size=self.win_size
                    )
                else:
                    C = sliding_window_ref_aware(
                        C,
                        denoiser=self.denoiser,
                        p_module=self.p_module,
                        win_size=self.win_size
                    )

            # --------------------------------------------------
            # Gradient control
            # --------------------------------------------------
            if not fine_tune:
                C = C.detach()

        return C

'''
class UnrolledBatchMLEM_vanilla(nn.Module):
    def __init__(
        self,
        denoiser: nn.Module,
        n_iter: int = 10,
        beta: float = 0.1,
        delta: float = 0.01,
        warmup_iters: int = 3,
        device = 'cuda'
    ):
        super().__init__()
        self.denoiser = denoiser
        self.n_iter = n_iter
        self.beta = beta
        self.delta = delta
        self.device = device
        self.warmup_iters = warmup_iters

    def move_data_to_device(
        self,
        C_init,    # (n_ref, B, H, W)
        atten,     # (n_angle, B, H, W)
        em_cs,     # (n_angle, n_ref)
        theta,     # (n_angle,)
        I,         # (n_angle, B, W)
    ):
        if not torch.is_tensor(C_init):
            C_init = torch.from_numpy(C_init).float()
        if not torch.is_tensor(atten):
            atten = torch.from_numpy(atten).float()
        if not torch.is_tensor(em_cs):
            em_cs = torch.from_numpy(em_cs).float()
        if not torch.is_tensor(theta):
            theta = torch.from_numpy(theta).float()
        if not torch.is_tensor(I):
            I = torch.from_numpy(I).float()
        
        C_init.to(self.device)
        atten.to(self.device)
        em_cs.to(self.device)
        theta.to(self.device)
        I.to(self.device)

        self.C_init = C_init
        self.atten = atten
        self.em_cs = em_cs
        self.theta = theta
        self.I = I

    def forward(self, fine_tune: bool = True):
        # ---- safe starting point (no inplace) ----
        C = self.C_init.clone()

        for k in range(self.n_iter):
            # --------------------------------------------------
            # Physics update (1 MLEM step, autograd-enabled)
            # --------------------------------------------------
            C = FL.mlem_cuda_batch(
                C,
                self.atten,
                self.em_cs,
                self.theta,
                self.I,
                1,
                beta=self.beta,
                delta=self.delta
            )

            # --------------------------------------------------
            # 3D-UNet denoising (after warmup)
            # --------------------------------------------------
            if k >= self.warmup_iters:
                for i in range(len(C)):
                    C[i] = self.denoiser()

            # --------------------------------------------------
            # Gradient control
            # --------------------------------------------------
            if not fine_tune:
                C = C.detach()

        return C
####################################
## train step (single batch) 
####################################

def UnrolledBatchMLEM_train_step(
    model,                 # UnrolledBatchMLEM
    batch,                 # dict with tensors
    optimizer,
    device,
    fine_tune=True,
    grad_clip=1.0
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    # ----------------------------
    # Move data
    # ----------------------------
    C_init = batch["C_init"].to(device)   # (n_ref, B, H, W)
    atten  = batch["atten"].to(device)    # (n_angle, B, H, W)
    em_cs  = batch["em_cs"].to(device)    # (n_angle, n_ref)
    theta  = batch["theta"].to(device)    # (n_angle,)
    I      = batch["I"].to(device)         # (n_angle, B, W)

    # ----------------------------
    # Unrolled reconstruction
    # ----------------------------
    C = model(
        C_init,
        atten,
        em_cs,
        theta,
        I,
        fine_tune=fine_tune
    )

    # ----------------------------
    # Physics-consistent loss
    # ----------------------------
    Pf = forward_emission_batch_autograd(
        atten,
        C,
        em_cs,
        theta
    )

    loss = poisson_loss(Pf, I)

    # ----------------------------
    # Backprop
    # ----------------------------
    loss.backward()

    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip
        )

    optimizer.step()

    return {
        "loss": loss.item(),
        "C_mean": C.mean().item(),
        "C_max": C.max().item(),
    }


####################################
## validation (single batch) 
####################################

@torch.no_grad()
def validation_step(
    model,
    batch,
    device
):
    model.eval()

    C = model(
        batch["C_init"].to(device),
        batch["atten"].to(device),
        batch["em_cs"].to(device),
        batch["theta"].to(device),
        batch["I"].to(device),
        fine_tune=False     # 🚫 no gradient graph
    )

    Pf = forward_emission_batch_autograd(
        batch["atten"].to(device),
        C,
        batch["em_cs"].to(device),
        batch["theta"].to(device)
    )

    loss = poisson_loss(Pf, batch["I"].to(device))

    return loss.item()





## full train loop

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    n_epochs=50,
    warmup_epochs=5,
    log_every=10
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "p_values": []
    }

    for epoch in range(n_epochs):

        fine_tune = epoch >= warmup_epochs
        epoch_loss = 0.0

        for it, batch in enumerate(train_loader):

            stats = train_step(
                model,
                batch,
                optimizer,
                device,
                fine_tune=fine_tune
            )

            epoch_loss += stats["loss"]

            if it % log_every == 0:
                print(
                    f"[Epoch {epoch:03d} | Iter {it:04d}] "
                    f"loss={stats['loss']:.4e} "
                    f"mean(C)={stats['C_mean']:.3e}"
                )

        epoch_loss /= len(train_loader)
        history["train_loss"].append(epoch_loss)

        # ----------------------------
        # Validation
        # ----------------------------
        if val_loader is not None:
            val_loss = validation_step(
                model,
                next(iter(val_loader)),
                device
            )
            history["val_loss"].append(val_loss)
            print(f"  ↳ val_loss={val_loss:.4e}")

        # ----------------------------
        # Track adaptive percentile
        # ----------------------------
        if hasattr(model.p_module, "p"):
            history["p_values"].append(
                model.p_module.p.detach().cpu().numpy()
            )

    return history



###################################
## define the optimizer
###################################
optimizer = torch.optim.Adam(
    [
        {"params": model.denoiser.parameters(), "lr": 1e-4},
        {"params": model.p_module.parameters(), "lr": 5e-4},
    ],
    betas=(0.9, 0.999)
)





## trainin mlem+3DCNN example:


denoiser = RefAware3DCNN(n_ref=2, in_channels=2, out_channels=2).cuda()

# Define unrolled batched MLEM model
unrolled_mlem = UnrolledBatchMLEM(
    denoiser=denoiser,
    n_iter=10,
    win_size=5,
    beta=0.1,
    delta=0.01,
    warmup_iters=3
).cuda()
unrolled_mlem.use_checkpoint = True

# Optimizer (only for CNN parameters)
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-4)

# Dummy input data (replace with real dataset)
C_init = torch.rand(2, 8, 64, 64).cuda()       # (n_ref, B, H, W)
atten  = torch.rand(16, 8, 64, 64).cuda()      # (n_angle, B, H, W)
em_cs  = torch.rand(16, 2).cuda()              # (n_angle, n_ref)
theta  = torch.linspace(0, 3.14, 16).cuda()    # (n_angle,)
I      = torch.rand(16, 8, 64).cuda()          # (n_angle, B, W)

n_epoch = 5
# Training loop
for epoch in range(n_epochs):
    optimizer.zero_grad()
    
    C_rec = unrolled_mlem(C_init, atten, em_cs, theta, I, fine_tune=True)
    Pf = forward_emission_batch(atten, C_rec, em_cs, theta)
    Pf.clamp_min_(1e-6)
    
    loss = poisson_loss(Pf, I)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Poisson loss: {loss.item():.6f}")

import torch.nn.functional as F
import torch
import torch.nn as nn

#####################################################
################## 3D UNet  #########################
#####################################################



class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.conv = ConvBlock3D(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle odd sizes safely
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class RefAware3DUNet(nn.Module):
    def __init__(
            self,
            n_ref,
            emb_dim=8,
            base_ch=32
    ):
        super().__init__()

        self.n_ref = n_ref
        self.emb = nn.Embedding(n_ref, emb_dim)

        self.alpha = nn.Parameter(torch.tensor(0.05))  # safer than 0.1
        in_ch = 1 + emb_dim

        # Encoder
        self.inc = ConvBlock3D(in_ch, base_ch)
        self.down1 = Down3D(base_ch, base_ch * 2)
        self.down2 = Down3D(base_ch * 2, base_ch * 4)

        # Bottleneck
        self.mid = ConvBlock3D(base_ch * 4, base_ch * 4)

        # Decoder
        self.up1 = Up3D(base_ch * 4, base_ch * 2)
        self.up2 = Up3D(base_ch * 2, base_ch)

        # Output
        self.outc = nn.Conv3d(base_ch, 1, 1)

    def freeze_encoder(self):
        """Freeze encoder (Down path)"""
        for m in [self.inc, self.down1, self.down2]:
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        for m in [self.inc, self.down1, self.down2]:
            for p in m.parameters():
                p.requires_grad = True

    def freeze_embedding(self):
        for p in self.emb.parameters():
            p.requires_grad = False

    def forward(self, x, ref_idx):
        """
        x:       (B, 1, D, H, W)
        ref_idx: scalar or (B,)
        """
        B, _, D, H, W = x.shape
        x0 = x

        # -------- reference embedding --------
        emb = self.emb(ref_idx)  # (B, E)
        emb = emb.view(B, -1, 1, 1, 1)
        emb = emb.expand(-1, -1, D, H, W)

        x = torch.cat([x, emb], dim=1)

        # -------- U-Net --------
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.mid(x3)

        x = self.up1(x, x2)
        x = self.up2(x, x1)

        res = self.outc(x)

        # -------- residual output --------
        alpha = torch.clamp(self.alpha, 0.0, 0.1)
        return x0 + alpha * res

###################### END 3D U-Net #####################

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

################## Unrolled MLEM ####################
class UnrolledBatchMLEM_vanilla(nn.Module):
    def __init__(
            self,
            denoiser: nn.Module,
            n_iter: int = 10,
            beta: float = 0.1,
            delta: float = 0.01,
            warmup_iters: int = 3,
            device='cuda'
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
            C_init,  # (n_ref, B, H, W)
            atten,  # (n_angle, B, H, W)
            em_cs,  # (n_angle, n_ref)
            theta,  # (n_angle,)
            I,  # (n_angle, B, W)
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
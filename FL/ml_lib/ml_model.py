import torch.nn.functional as F
import torch
import torch.nn as nn
import functools
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

    def forward(self, x, ref_idx=None):
        """
        x:       (B, 1, D, H, W)
        ref_idx: scalar or (B,)
        """
        B, _, D, H, W = x.shape
        x0 = x
        if ref_idx is None:
            ref_idx = torch.zeros(B, dtype=torch.long, device=x.device)
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
        #alpha = torch.clamp(self.alpha, 0.0, 0.1)
        alpha = self.alpha
        return x0 + alpha * res

###################### END 3D U-Net #####################


##################### general 3D U-Net ##################
class RefAware3DUNet_general(nn.Module):
    def __init__(
        self,
        n_ref,
        emb_dim=8,
        base_ch=32,
        n_levels=2   # <---- NEW: number of down/up stages
    ):
        super().__init__()

        assert n_levels >= 1, "Need at least 1 level"

        self.n_ref = n_ref
        self.emb = nn.Embedding(n_ref, emb_dim)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        in_ch = 1 + emb_dim

        # -----------------------------
        # Encoder path
        # -----------------------------
        self.inc = ConvBlock3D(in_ch, base_ch)

        self.down = nn.ModuleList()
        ch = base_ch

        for i in range(n_levels):
            self.down.append(
                Down3D(ch, ch * 2)
            )
            ch *= 2

        # -----------------------------
        # Bottleneck
        # -----------------------------
        self.mid = ConvBlock3D(ch, ch)

        # -----------------------------
        # Decoder path
        # -----------------------------
        self.up = nn.ModuleList()

        for i in range(n_levels):
            self.up.append(Up3D(ch, ch // 2))
            ch //= 2

        # -----------------------------
        # Output layer
        # -----------------------------
        self.outc = nn.Conv3d(base_ch, 1, kernel_size=1)
        self.n_levels = n_levels

    # -------------------------------------------------
    # Utilities to freeze / unfreeze encoder
    # -------------------------------------------------
    def freeze_encoder(self):
        for m in [self.inc] + list(self.down):
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_encoder(self):
        for m in [self.inc] + list(self.down):
            for p in m.parameters():
                p.requires_grad = True

    def freeze_embedding(self):
        for p in self.emb.parameters():
            p.requires_grad = False

    # -------------------------------------------------
    # Forward
    # -------------------------------------------------
    def forward(self, x, ref_idx=None):
        """
        x:       (B, 1, D, H, W)
        ref_idx: scalar or (B,)
        """
        B, _, D, H, W = x.shape
        x0 = x
        if ref_idx is None:
            ref_idx = torch.zeros(B, dtype=torch.long, device=x.device)
        # ----- reference embedding -----
        emb = self.emb(ref_idx)           # (B, E)
        emb = emb.view(B, -1, 1, 1, 1)
        emb = emb.expand(-1, -1, D, H, W)

        x = torch.cat([x, emb], dim=1)

        # ----- Encoder -----
        skips = []

        x = self.inc(x)
        skips.append(x)

        for down in self.down:
            x = down(x)
            skips.append(x)

        # Last element in skips corresponds to bottleneck input
        skips = skips[:-1]   # remove deepest (not used for skip)

        # ----- Bottleneck -----
        x = self.mid(x)

        # ----- Decoder -----
        for i, up in enumerate(self.up):
            skip = skips[-(i + 1)]
            x = up(x, skip)

        # ----- Output -----
        res = self.outc(x)

        #alpha = torch.clamp(self.alpha, 0.0, 0.1)
        alpha = self.alpha

        return x0 + alpha * res

    ##############################################################

class RefAware3DCNN(nn.Module):
    def __init__(self, n_ref=1, emb_dim=8, hidden=32):
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

################ 2D RRDB ##################
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class ResidualDenseBlock_3C(nn.Module):
    def __init__(self, nf=32, gc=16, bias=True, kernel_size=3):
        super(ResidualDenseBlock_3C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        #self.conv11 = nn.Conv2d(nf, int(gc/2), 3, stride=1, dilation=1, padding='same', bias=bias)
        self.conv1 = nn.Conv2d(nf, gc, kernel_size, stride=1, padding='same', bias=bias, padding_mode='replicate')
        self.conv2 = nn.Conv2d(nf + gc, gc, kernel_size, stride=1, padding='same', bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, nf, kernel_size, 1, 'same', bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        s = {}
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        return x3 * 0.4 + x

class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32, kernel_size=3):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_3C(nf, gc, kernel_size=kernel_size)
        self.RDB2 = ResidualDenseBlock_3C(nf, gc, kernel_size=kernel_size)
        self.RDB3 = ResidualDenseBlock_3C(nf, gc, kernel_size=kernel_size)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, padding_mode='zeros', kernel_size=3):
        self.model_config = {}
        self.model_config['in_nc'] = in_nc
        self.model_config['out_nc'] = out_nc
        self.model_config['nf'] = nf
        self.model_config['nb'] = nb
        self.model_config['gc'] = gc
        self.model_config['padding_mode'] = padding_mode
        self.model_config['kernel_size'] = kernel_size
        self.model_config['model_structure'] = 'RRDBNet'

        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc, kernel_size=kernel_size)

        self.conv_first_1 = nn.Conv2d(in_nc, int(nf/2), kernel_size, 1, 1, padding_mode=padding_mode, bias=True)
        self.conv_first_2 = nn.Conv2d(in_nc, int(nf/2), kernel_size, stride=1, dilation=1, padding='same', padding_mode=padding_mode, bias=True)
        self.conv_first = nn.Conv2d(in_nc, nf, kernel_size, 1, 1, padding_mode=padding_mode, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, kernel_size, 1, 1,  padding_mode=padding_mode, bias=True)
        #### upsampling
        #self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, kernel_size, 1, 1, padding_mode=padding_mode, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, kernel_size, 1, 1, padding_mode=padding_mode, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk # this is original one used
        out = self.conv_last(self.lrelu(self.HRconv(fea))) # the following is added
        out = torch.nn.functional.softplus(out)
        return out

################ 2D UNet ###################
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        # Down part
        for feature in features:
            self.downs.append(UNetBlock(in_channels, feature))
            in_channels = feature
        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(UNetBlock(feature*2, feature))
        self.bottleneck = UNetBlock(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)
        return self.final_conv(x)
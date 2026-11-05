import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Utilities
# ============================================================

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)



# ============================================================
# Residual Dense Block
# ============================================================

class ResidualDenseBlock3C(nn.Module):
    """
    Residual Dense Block.

    Same basic structure as your original implementation,
    but dilation is configurable.
    """

    def __init__(
        self,
        nf=32,
        gc=16,
        kernel_size=3,
        dilation=1,
        bias=True,
        padding_mode="replicate",
    ):
        super().__init__()

        assert kernel_size % 2 == 1

        padding = dilation * (kernel_size // 2)

        self.conv1 = nn.Conv2d(
            nf,
            gc,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.conv2 = nn.Conv2d(
            nf + gc,
            gc,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.conv3 = nn.Conv2d(
            nf + 2 * gc,
            nf,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.lrelu = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True,
        )

    def forward(self, x):

        x1 = self.lrelu(self.conv1(x))

        x2 = self.lrelu(
            self.conv2(
                torch.cat((x, x1), dim=1)
            )
        )

        x3 = self.lrelu(
            self.conv3(
                torch.cat((x, x1, x2), dim=1)
            )
        )

        # Residual scaling follows your original RDB.
        return x + 0.4 * x3


# ============================================================
# RRDB with dilation
# ============================================================

class RRDB_dilation(nn.Module):
    """
    Residual-in-Residual Dense Block.

    dilation=1 -> normal RRDB
    dilation=2 -> dilated RRDB
    """

    def __init__(
        self,
        nf,
        gc=32,
        kernel_size=3,
        dilation=1,
    ):
        super().__init__()

        self.RDB1 = ResidualDenseBlock3C(
            nf=nf,
            gc=gc,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.RDB2 = ResidualDenseBlock3C(
            nf=nf,
            gc=gc,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.RDB3 = ResidualDenseBlock3C(
            nf=nf,
            gc=gc,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):

        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        return x + 0.2 * out


# ============================================================
# Low-frequency branch
# ============================================================

class LowFrequencyBranch(nn.Module):
    """
    Explicit low-frequency branch.

    Idea:
        1. Low-pass filter the feature map using average pooling.
        2. Downsample by 2.
        3. Process at low spatial resolution.
        4. Upsample to original resolution.

    This branch is intended to capture large-scale / slowly
    varying structures and broad missing-angle bias.
    """

    def __init__(
        self,
        nf=16,
        num_rrdb=1,
        gc=32,
    ):
        super().__init__()

        # Average pooling explicitly suppresses high frequencies.
        self.lowpass = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv_in = nn.Conv2d(
            nf,
            nf,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        rrdb_block = functools.partial(
            RRDB_dilation,
            nf=nf,
            gc=gc,
            kernel_size=3,
            dilation=1,
        )

        self.rrdb = make_layer(
            rrdb_block,
            num_rrdb,
        )

        self.conv_out = nn.Conv2d(
            nf,
            nf,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

        self.act = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Explicit low-pass + downsample
        # ----------------------------------------------------
        y = self.lowpass(x)

        y = self.act(
            self.conv_in(y)
        )

        # ----------------------------------------------------
        # Process coarse-scale structure
        # ----------------------------------------------------
        residual = self.rrdb(y)

        y = y + self.conv_out(residual)

        # ----------------------------------------------------
        # Restore original resolution
        # ----------------------------------------------------
        y = F.interpolate(
            y,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return y


# ============================================================
# Dilated branch
# ============================================================

class DilatedBranch(nn.Module):
    """
    Dilated branch operating at half resolution.

    The downsampling increases the physical receptive field
    even further, while dilation gives additional context.
    """

    def __init__(
        self,
        nf=16,
        num_rrdb=2,
        gc=32,
        dilation=2,
    ):
        super().__init__()

        # Learnable downsampling.
        self.down = nn.Conv2d(
            nf,
            nf,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
        )

        self.act = nn.LeakyReLU(
            negative_slope=0.2,
            inplace=True,
        )

        rrdb_block = functools.partial(
            RRDB_dilation,
            nf=nf,
            gc=gc,
            kernel_size=3,
            dilation=dilation,
        )

        self.rrdb = make_layer(
            rrdb_block,
            num_rrdb,
        )

        # Additional dilated convolution.
        self.dilated_conv = nn.Conv2d(
            nf,
            nf,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=True,
        )

        self.proj = nn.Conv2d(
            nf,
            nf,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    def forward(self, x):

        # ----------------------------------------------------
        # Downsample
        # ----------------------------------------------------
        y = self.down(x)
        y = self.act(y)

        # ----------------------------------------------------
        # Dilated processing
        # ----------------------------------------------------
        residual = self.rrdb(y)

        residual = self.act(
            self.dilated_conv(residual)
        )

        y = y + residual

        # ----------------------------------------------------
        # Upsample
        # ----------------------------------------------------
        y = F.interpolate(
            y,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        y = self.proj(y)

        return y


# ============================================================
# Global branch
# ============================================================

class GlobalMultiScaleBranch(nn.Module):
    """
    Global branch containing TWO explicitly different pathways:

             input
               |
        +------+------+
        |             |
      low-pass      dilated
        |             |
      coarse        coarse
      RRDB           RRDB
        |             |
        +------+------+
               |
             fusion
               |
             output

    The low-frequency path is intended to capture broad
    intensity / object-scale structure.

    The dilated path is intended to capture long-range
    structured artifacts such as missing-angle streaks.
    """

    def __init__(
        self,
        nf=16,
        gc=32,
        lowres_rrdb=1,
        dilated_rrdb=2,
        dilation=2,
    ):
        super().__init__()

        self.low_freq = LowFrequencyBranch(
            nf=nf,
            gc=gc,
            num_rrdb=lowres_rrdb,
        )

        self.dilated = DilatedBranch(
            nf=nf,
            gc=gc,
            num_rrdb=dilated_rrdb,
            dilation=dilation,
        )

        # Fuse low-frequency and dilated features.
        self.fusion = nn.Sequential(

            nn.Conv2d(
                2 * nf,
                nf,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),

            nn.Conv2d(
                nf,
                nf,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
        )

    def forward(self, x):

        low = self.low_freq(x)

        dilated = self.dilated(x)

        combined = torch.cat(
            [low, dilated],
            dim=1,
        )

        global_feat = self.fusion(
            combined
        )

        # Residual global representation.
        return x + global_feat


# ============================================================
# Final model
# ============================================================

class RRDBNetMultiScaleResidual(nn.Module):
    """
    Multi-scale residual RRDB network.

    Architecture:

                         Local branch
                      RRDB x N
                           |
                           |
    input --> features ---+--------------------+
                           |                    |
                           |             Global branch
                           |                    |
                           |             +------+------+
                           |             |             |
                           |        low-frequency   dilated
                           |             |             |
                           |             +------+------+
                           |                    |
                           +--------- fusion ---+
                                        |
                                artifact residual
                                        |
                           Iout = Iinput + alpha * R

    Parameters roughly compatible with your current model:

        RRDBNetMultiScaleResidual(
            1, 1, 16, 4, 32
        )
    """

    def __init__(
        self,
        in_nc=1,
        out_nc=1,
        nf=16,
        nb=4,
        gc=32,
        padding_mode="zeros",
        kernel_size=3,

        # Global branch
        lowres_rrdb=3,
        dilated_rrdb=3,
        dilation=3,

        # Residual scale
        residual_scale_init=0.2,
    ):
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.gc = gc
        self.padding_mode = padding_mode
        self.kernel_size = kernel_size
        self.lowres_rrdb = lowres_rrdb
        self.dilated_rrdb = dilated_rrdb
        self.dilation = dilation
        self.residual_scale_init = residual_scale_init
        self.model_structure = 'RRDBNetMultiScaleResidual'
        self.model_config = {}
        self.model_config['in_nc'] = in_nc
        self.model_config['out_nc'] = out_nc
        self.model_config['nf'] = nf
        self.model_config['nb'] = nb
        self.model_config['gc'] = gc
        self.model_config['padding_mode'] = padding_mode
        self.model_config['kernel_size'] = kernel_size
        self.model_config['lowres_rrdb'] = lowres_rrdb
        self.model_config['dilated_rrdb'] = dilated_rrdb
        self.model_config['dilation'] = dilation
        self.model_config['residual_scale_init'] = residual_scale_init
        self.model_config['model_structure'] = self.model_structure


        super().__init__()

        assert kernel_size % 2 == 1

        padding = kernel_size // 2

        # ====================================================
        # Input feature extraction
        # ====================================================

        self.conv_first = nn.Conv2d(
            in_nc,
            nf,
            kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=True,
        )

        # ====================================================
        # Local branch
        # ====================================================

        rrdb_block_local = functools.partial(
            RRDB_dilation,
            nf=nf,
            gc=gc,
            kernel_size=kernel_size,
            dilation=1,
        )

        self.local_rrdb = make_layer(
            rrdb_block_local,
            nb,
        )

        self.local_trunk_conv = nn.Conv2d(
            nf,
            nf,
            kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            bias=True,
        )

        # ====================================================
        # Global branch
        # ====================================================

        self.global_branch = GlobalMultiScaleBranch(
            nf=nf,
            gc=gc,
            lowres_rrdb=lowres_rrdb,
            dilated_rrdb=dilated_rrdb,
            dilation=dilation,
        )

        # ====================================================
        # Local + global fusion
        # ====================================================

        self.fusion = nn.Sequential(

            # local = nf
            # global = nf
            # total = 2*nf
            nn.Conv2d(
                2 * nf,
                nf,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),

            nn.Conv2d(
                nf,
                nf,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),
        )

        # ====================================================
        # Residual prediction head
        # ====================================================

        self.residual_head = nn.Sequential(

            nn.Conv2d(
                nf,
                nf,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=padding_mode,
                bias=True,
            ),

            nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True,
            ),

            nn.Conv2d(
                nf,
                out_nc,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                padding_mode=padding_mode,
                bias=True,
            ),
        )

        # ====================================================
        # Learnable residual scaling
        # ====================================================

        self.alpha = nn.Parameter(
            torch.tensor(float(residual_scale_init))
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # Original corrupted reconstruction
        # ----------------------------------------------------

        x0 = x

        # ----------------------------------------------------
        # Shared feature extraction
        # ----------------------------------------------------

        fea = self.conv_first(x)

        # ----------------------------------------------------
        # Local branch
        # ----------------------------------------------------

        local = self.local_rrdb(fea)

        local = self.local_trunk_conv(
            local
        )

        local = fea + local

        # ----------------------------------------------------
        # Global branch
        # ----------------------------------------------------

        global_feat = self.global_branch(
            fea
        )

        # ----------------------------------------------------
        # Fuse local + global
        # ----------------------------------------------------

        fused = torch.cat(
            [local, global_feat],
            dim=1,
        )

        fused = self.fusion(
            fused
        )

        # Keep shared features through residual fusion.
        fused = fea + fused

        # ----------------------------------------------------
        # Predict artifact residual
        # ----------------------------------------------------

        artifact_residual = self.residual_head(
            fused
        )

        # ----------------------------------------------------
        # Reconstruct corrected image
        # ----------------------------------------------------

        output = (
            x0
            + self.alpha * artifact_residual
        )

        return output
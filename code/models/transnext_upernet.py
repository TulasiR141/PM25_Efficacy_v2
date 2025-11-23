# --------------------------------------------------------
# TransNeXt-like Encoder + UPerNet Decoder (Binary Segmentation)
# Fully fixed: PPM GroupNorm, FPN alignment, correct channels
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASIC CONV BLOCK
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================
# SIMPLE HIERARCHICAL TRANSFORMER-LIKE ENCODER
# Produces: C1, C2, C3, C4
# ============================================================

class PatchEncoder(nn.Module):
    def __init__(self, in_channels=3, dims=[64, 128, 256, 512]):
        super().__init__()

        self.stem = ConvBlock(in_channels, dims[0], k=7, s=2, p=3)
        self.stage1 = nn.Sequential(ConvBlock(dims[0], dims[0]), ConvBlock(dims[0], dims[0]))

        self.down1 = ConvBlock(dims[0], dims[1], k=3, s=2, p=1)
        self.stage2 = nn.Sequential(ConvBlock(dims[1], dims[1]), ConvBlock(dims[1], dims[1]))

        self.down2 = ConvBlock(dims[1], dims[2], k=3, s=2, p=1)
        self.stage3 = nn.Sequential(ConvBlock(dims[2], dims[2]), ConvBlock(dims[2], dims[2]))

        self.down3 = ConvBlock(dims[2], dims[3], k=3, s=2, p=1)
        self.stage4 = nn.Sequential(ConvBlock(dims[3], dims[3]), ConvBlock(dims[3], dims[3]))

        self.out_channels = dims

    def forward(self, x):
        c1 = self.stage1(self.stem(x))     # 1/2 resolution
        c2 = self.stage2(self.down1(c1))   # 1/4
        c3 = self.stage3(self.down2(c2))   # 1/8
        c4 = self.stage4(self.down3(c3))   # 1/16
        return [c1, c2, c3, c4]


# ============================================================
# PYRAMID POOLING MODULE (PPM) — FIXED WITH GROUPNORM
# ============================================================

class PPM(nn.Module):
    def __init__(self, in_ch, out_ch=256, pool_sizes=[1, 2, 3, 6]):
        super().__init__()

        self.stages = nn.ModuleList()
        for ps in pool_sizes:
            self.stages.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_ch, out_ch, 1, bias=False),
                    nn.GroupNorm(32, out_ch),   # FIX: GN works for 1×1 features
                    nn.ReLU(inplace=True)
                )
            )

        total_ch = in_ch + len(pool_sizes) * out_ch

        self.bottleneck = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),         # FIX: GN
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        outs = [x]

        for stage in self.stages:
            pooled = stage(x)
            outs.append(F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False))

        return self.bottleneck(torch.cat(outs, dim=1))


# ============================================================
# UPerNet DECODER — FIXED CHANNELS + SAFE ALIGNMENT
# ============================================================

class UPerNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        c1, c2, c3, c4 = in_channels

        self.ppm = PPM(c4, out_ch=256)

        self.lat3 = nn.Conv2d(c3, 256, 1)
        self.lat2 = nn.Conv2d(c2, 256, 1)
        self.lat1 = nn.Conv2d(c1, 256, 1)

        self.smooth3 = ConvBlock(256, 256)
        self.smooth2 = ConvBlock(256, 256)
        self.smooth1 = ConvBlock(256, 256)

        # Final fusion of 4 × 256 = 1024 → 256 → output
        self.final = nn.Sequential(
            ConvBlock(1024, 256),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, feats):
        c1, c2, c3, c4 = feats

        p4 = self.ppm(c4)

        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="bilinear"))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode="bilinear"))
        p1 = self.smooth1(self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode="bilinear"))

        # Safe alignment before concatenation
        h, w = c1.shape[2:]
        p4 = F.interpolate(p4, size=(h, w), mode="bilinear")
        p3 = F.interpolate(p3, size=(h, w), mode="bilinear")
        p2 = F.interpolate(p2, size=(h, w), mode="bilinear")

        fused = torch.cat([p4, p3, p2, p1], dim=1)
        return self.final(fused)


# ============================================================
# FULL MODEL
# ============================================================

class TransNeXtUperNet(nn.Module):
    def __init__(self, in_size=512, in_channels=3, out_channels=1, variant="tiny"):
        super().__init__()

        if variant == "tiny":
            dims = [64, 128, 256, 512]
        elif variant == "small":
            dims = [96, 192, 384, 768]
        elif variant == "base":
            dims = [128, 256, 512, 1024]
        else:
            raise ValueError("variant must be tiny|small|base")

        self.encoder = PatchEncoder(in_channels=in_channels, dims=dims)
        self.decoder = UPerNetDecoder(dims, out_channels=out_channels)

    def forward(self, x):
        feats = self.encoder(x)
        seg = self.decoder(feats)
        return F.interpolate(seg, size=x.shape[2:], mode="bilinear", align_corners=False)


# ============================================================
# WRAPPER CLASSES
# ============================================================

class TransNeXtUperNet_Tiny(TransNeXtUperNet):
    def __init__(self, **kwargs):
        super().__init__(variant="tiny", **kwargs)


class TransNeXtUperNet_Small(TransNeXtUperNet):
    def __init__(self, **kwargs):
        super().__init__(variant="small", **kwargs)


class TransNeXtUperNet_Base(TransNeXtUperNet):
    def __init__(self, **kwargs):
        super().__init__(variant="base", **kwargs)

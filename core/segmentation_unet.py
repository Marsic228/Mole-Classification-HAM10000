import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.bridge = conv_block(64, 128)

        self.dec3 = conv_block(128 + 64, 64)
        self.dec2 = conv_block(64 + 32, 32)
        self.dec1 = conv_block(32 + 16, 16)

        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        h = self.bridge(F.max_pool2d(e3, 2))

        h = self.dec3(torch.cat([
            F.interpolate(h, size=e3.shape[-2:],
                          mode="bilinear", align_corners=False),
            e3
        ], dim=1))

        h = self.dec2(torch.cat([
            F.interpolate(h, size=e2.shape[-2:],
                          mode="bilinear", align_corners=False),
            e2
        ], dim=1))

        h = self.dec1(torch.cat([
            F.interpolate(h, size=e1.shape[-2:],
                          mode="bilinear", align_corners=False),
            e1
        ], dim=1))

        return self.head(h)
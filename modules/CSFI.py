import torch.nn as nn
import torch
from modules.CSM import CSM


class CSFI(nn.Module):
    def __init__(self, channel):
        super(CSFI, self).__init__()

        self.low_up1 = nn.Sequential(*[
            nn.ConvTranspose2d(channel * 4, channel * 2, 4, 2, 1),
            nn.GELU(),
        ])
        self.low_up2 = nn.Sequential(*[
            nn.ConvTranspose2d(channel * 2, channel, 4, 2, 1),
            nn.GELU(),
        ])

        self.mid_up = nn.Sequential(*[
            nn.ConvTranspose2d(channel * 2, channel, 4, 2, 1),
            nn.GELU(),
        ])
        self.mid_down = nn.Sequential(*[
            nn.Conv2d(channel * 2, channel * 4, 3, 2, 1),
            nn.GELU(),
        ])

        self.high_down1 = nn.Sequential(*[
            nn.Conv2d(channel, channel * 2, 3, 2, 1),
            nn.GELU(),
        ])
        self.high_down2 = nn.Sequential(*[
            nn.Conv2d(channel * 2, channel * 4, 3, 2, 1),
            nn.GELU(),
        ])

        self.conv_low = nn.Conv2d(channel * 12, channel * 4, 3, 1, 1)
        self.conv_mid = nn.Conv2d(channel * 6, channel * 2, 3, 1, 1)
        self.conv_high = nn.Conv2d(channel * 3, channel * 1, 3, 1, 1)

        self.CSM_low = CSM(channel * 4)
        self.CSM_mid = CSM(channel * 2)
        self.CSM_high = CSM(channel)

    def forward(self, scales):
        high, mid, low = scales[0], scales[1], scales[2]

        l2m = self.low_up1(low)
        l2h = self.low_up2(l2m)
        m2h = self.mid_up(mid)
        m2l = self.mid_down(mid)
        h2m = self.high_down1(high)
        h2l = self.high_down2(h2m)

        low = self.conv_low(torch.cat([low, m2l, h2l], 1))
        mid = self.conv_mid(torch.cat([l2m, mid, h2m], 1))
        high = self.conv_high(torch.cat([l2h, m2h, high], 1))

        low = self.CSM_low(low)
        mid = self.CSM_mid(mid)
        high = self.CSM_high(high)

        return [high, mid, low]

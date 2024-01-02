import torch
import torch.nn as nn
from modules.CSM import CSM


class Encoder(nn.Module):
    def __init__(self, channel):
        super(Encoder, self).__init__()
        self.channel_extend = nn.Sequential(*[
            nn.Conv2d(3, channel, 3, 1, 1),
            nn.GELU(),
        ])
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel) for _ in range(4)]),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 3, 2, 1),
            nn.Conv2d(channel * 2, channel * 2, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 2) for _ in range(3)]),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(channel * 2, channel * 4, 3, 2, 1),
            nn.Conv2d(channel * 4, channel * 4, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 4) for _ in range(2)]),
        )
        self.layer4 = nn.Conv2d(channel * 4, channel * 8, 3, 2, 1)

    def forward(self, x):
        x = self.channel_extend(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        scales = [x1, x2, x3, x4]
        return scales


class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(*[
            nn.ConvTranspose2d(channel * 8, channel * 4, 4, 2, 1),
            nn.GELU(),
        ])
        self.layer2 = nn.Sequential(*[
            nn.Conv2d(channel * 8, channel * 4, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 4) for _ in range(2)]),
            nn.ConvTranspose2d(channel * 4, channel * 2, 4, 2, 1),
            nn.GELU(),
        ])
        self.layer3 = nn.Sequential(*[
            nn.Conv2d(channel * 4, channel * 2, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 2) for _ in range(3)]),
            nn.ConvTranspose2d(channel * 2, channel, 4, 2, 1),
            nn.GELU(),
        ])
        self.layer4 = nn.Sequential(*[
            nn.Conv2d(channel * 2, channel, 3, 1, 1),
            nn.GELU(),
            nn.Sequential(*[CSM(channel * 1) for _ in range(4)]),
        ])
        self.final_CSM = nn.Sequential(*[CSM(channel) for _ in range(4)])
        self.channel_reduction = nn.Conv2d(channel, 3, 3, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = x[3], x[2], x[1], x[0]
        x1 = self.layer1(x1)
        x2 = self.layer2(torch.cat([x2, x1], dim=1))
        x3 = self.layer3(torch.cat([x3, x2], dim=1))
        x4 = self.layer4(torch.cat([x4, x3], dim=1))
        x4 = self.final_CSM(x4)
        x4 = self.channel_reduction(x4)
        return x4

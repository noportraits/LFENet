import torch
import torch.nn as nn
from modules.IEM import IEM
from modules.Encoder_Decoder import Encoder, Decoder
from modules.CVMI import CVMI
from modules.CSFI import CSFI
from modules.CSM import CSM


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.channel = 16
        self.IEM = IEM()
        self.Encoder = Encoder(self.channel)
        self.CVMI = nn.ModuleList([
            CVMI(self.channel),
            CVMI(self.channel * 2),
            CVMI(self.channel * 4),
        ])
        self.CSFI = CSFI(self.channel)
        self.scale_4 = nn.Sequential(*[CSM(self.channel * 8) for _ in range(1)])
        self.Decoder = Decoder(self.channel)

    def forward(self, light_l, light_r, low_l, low_r):
        init_enhance_l, init_enhance_r = self.IEM(light_l, low_l), self.IEM(light_r, low_r)
        scales_l, scales_r = self.Encoder(init_enhance_l), self.Encoder(init_enhance_r)
        scales_l[0], scales_r[0] = self.CVMI[0](scales_l[0], scales_r[0])
        scales_l[1], scales_r[1] = self.CVMI[1](scales_l[1], scales_r[1])
        scales_l[2], scales_r[2] = self.CVMI[2](scales_l[2], scales_r[2])
        scales_l[0:3], scales_r[0:3] = self.CSFI(scales_l[0:3]), self.CSFI(scales_r[0:3])
        scales_l[3], scales_r[3] = self.scale_4(scales_l[3]), self.scale_4(scales_r[3])
        left, right = self.Decoder(scales_l), self.Decoder(scales_r)
        left, right = (torch.tanh(left) + 1) / 2, (torch.tanh(right) + 1) / 2
        return left, right

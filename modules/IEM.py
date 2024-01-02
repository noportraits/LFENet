import torch
import torch.nn as nn
from modules.CA import CA


class IEM(nn.Module):
    def __init__(self):
        super(IEM, self).__init__()
        self.CA = CA(channel=6)
        self.conv = nn.Conv2d(6, 3, 3, 1, 1)

    def forward(self, light_part, low_light):
        fusion = torch.cat([light_part, low_light], dim=1)
        fusion = self.CA(fusion)
        enhance = self.conv(fusion)
        return enhance

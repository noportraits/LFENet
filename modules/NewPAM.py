import torch.nn as nn
import torch
from modules.CSM import CSM


class NewPAM(nn.Module):

    def __init__(self, channels, height):
        super(NewPAM, self).__init__()
        self.scim = CSM(channels)
        self.height = height
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1) for _ in range(height)
        ])

    def forward(self, left, right):
        b, c, h, w = left.size()
        temp_left, temp_right = self.scim(left), self.scim(right)
        for i in range(self.height):
            right = self.conv_list[i](right + temp_left)
            left = self.conv_list[i](left + temp_right)
            temp_left = torch.cat(
                [temp_left, torch.zeros((b, c, h, 1), dtype=temp_left.dtype, device=temp_left.device)], dim=3)
            temp_left = temp_left[:, :, 1:]

            temp_right = torch.cat(
                [torch.zeros((b, c, h, 1), dtype=temp_right.dtype, device=temp_right.device), temp_right], dim=3)
            temp_right = temp_right[:, :, :, :-1]
        return left, right

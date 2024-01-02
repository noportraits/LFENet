import torch.nn as nn


class CA(nn.Module):
    def __init__(self, channel):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=1, padding=0, stride=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // 2, channel, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

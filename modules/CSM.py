import torch.nn as nn
import torch
from modules.CA import CA


class CSM(nn.Module):
    def __init__(self, channel):
        super(CSM, self).__init__()
        self.channel = channel
        self.SIM = nn.Sequential(
            LayerNorm2d(channel),
            nn.Conv2d(channel, channel, kernel_size=5, padding=2, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(channel // 2, channel, kernel_size=1, stride=1, bias=True),
        )
        self.CIM = nn.Sequential(
            LayerNorm2d(channel),
            CA(channel),
            nn.Conv2d(channel, channel * 8, kernel_size=1, stride=1, bias=True),
            SimpleGate(),
            nn.Conv2d(channel * 4, channel, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        y = self.SIM(x) + x
        y = self.CIM(y) + y
        return y


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2




import numpy as np
import torch
import torch.fft
import torch.nn as nn
from skimage.metrics import structural_similarity as compare_ssim


class LossFre(nn.Module):
    def __init__(self):
        super(LossFre, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, imgs, gts):
        imgs = torch.fft.rfftn(imgs, dim=(2, 3))
        _real = imgs.real
        _imag = imgs.imag
        imgs = torch.cat([_real, _imag], dim=1)
        gts = torch.fft.rfftn(gts, dim=(2, 3))
        _real = gts.real
        _imag = gts.imag
        gts = torch.cat([_real, _imag], dim=1)
        return self.criterion(imgs, gts)


class LossSpa(torch.nn.Module):
    def __init__(self):
        super(LossSpa, self).__init__()

    def tensor2image(self, tensor):
        tensor = tensor.squeeze(0)
        numpy_image = tensor.cpu().detach().numpy().astype(np.float32)
        numpy_image = np.transpose(numpy_image)
        numpy_image = (numpy_image * 255).astype(np.uint8)
        return numpy_image

    def forward(self, imageA, imageB):
        b, _, _, _ = imageA.shape
        ssim = 0
        for i in range(b):
            image1 = imageA[i, :, :, :]
            image2 = imageB[i, :, :, :]
            image1 = self.tensor2image(image1)
            image2 = self.tensor2image(image2)
            a = 1 - compare_ssim(image1, image2, win_size=11, channel_axis=2, data_range=255)
            ssim += a
        return ssim / b

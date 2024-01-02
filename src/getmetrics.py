import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def psnr_ssim(a, b):
    pre = tensor2cv(a)
    gt = tensor2cv(b)
    psnr, ssim = CALC(pre, gt)
    return psnr, ssim


def tensor2cv(tensor):

    numpy_image = tensor.cpu().detach().numpy()
    numpy_image = np.transpose(numpy_image)
    numpy_image = (numpy_image * 255).astype(np.uint8)
    # opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return numpy_image


def CALC(image1, image2):
    a = PSNR(image1, image2)
    b = SSIM(image1, image2)
    return a, b


def PSNR(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def SSIM(image1, image2):
    ssim_score = compare_ssim(image1, image2, win_size=5, channel_axis=2)
    return ssim_score

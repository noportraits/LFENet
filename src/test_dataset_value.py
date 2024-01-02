import math
import os
import cv2
import numpy as np
from skimage import metrics
from skimage.metrics import structural_similarity as ssim


def psnr_ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    psnr, ssim = CALC(a, b)
    return psnr, ssim


def psnr_a(img1, img2):
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    psnr = 20 * math.log10(255 / math.sqrt(mse))
    return psnr


def CALC(image1, image2):
    a = psnr_a(image1, image2)
    b = SSIM(image1, image2)
    return a, b


def PSNR(tar_img, prd_img):
    mse = np.mean((tar_img - prd_img) ** 2)
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


def SSIM(tar_img, prd_img):
    # Compute SSIM
    ssim_score = metrics.structural_similarity(tar_img, prd_img)
    return ssim_score


path1 = r"D:\mydataset\bestbignoise\holopix\pre\ours\right"
path2 = r"D:\mydataset\bestbignoise\holopix\test\normal\gt\right"
a = os.listdir(path1)
count = 0
psnr_total = 0
ssim_total = 0
for i in a:
    count += 1
    img = cv2.imread(path1 + "\\" + i)
    gt = cv2.imread(path2 + "\\" + i)
    psnr1, ssim1 = psnr_ssim(img, gt)
    print(i, ":", " psnr: ", psnr1, " ssim: ", ssim1)
    psnr_total += psnr1
    ssim_total += ssim1
print("total: ", psnr_total / count, ssim_total / count)
print(count)

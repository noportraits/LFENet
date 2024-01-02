import numpy as np
import torch
import cv2
import random

def save_img(img, img_path):
    enhanced_image = torch.squeeze(img, 0)
    enhanced_image = enhanced_image.permute(1, 2, 0)
    enhanced_image = np.asarray(enhanced_image.cpu())
    enhanced_image = enhanced_image * 255.0
    cv2.imwrite(img_path, enhanced_image)


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
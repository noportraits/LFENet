import torch
import numpy as np
import util
import os
import cv2
import argparse
from modules.Net import Net
from getmetrics import psnr_ssim


def lowlight(config, file_name, i):
    # 加载 light_l 图像
    light_l = cv2.imread(config.light_l + "\\" + file_name)
    light_l = torch.from_numpy(np.asarray(light_l)).float()
    light_l = light_l.permute(2, 0, 1)
    light_l = light_l.unsqueeze(0).cuda() / 255.0

    # 加载 light_r 图像
    light_r = cv2.imread(config.light_r + "\\" + file_name.replace("left", "right"))
    light_r = torch.from_numpy(np.asarray(light_r)).float()
    light_r = light_r.permute(2, 0, 1)
    light_r = light_r.unsqueeze(0).cuda() / 255.0

    # 加载 low_l 图像
    low_l = cv2.imread(config.low_l + "\\" + file_name)
    low_l = torch.from_numpy(np.asarray(low_l)).float()
    low_l = low_l.permute(2, 0, 1)
    low_l = low_l.unsqueeze(0).cuda() / 255.0

    # 加载 low_r 图像
    low_r = cv2.imread(config.low_r + "\\" + file_name.replace("left", "right"))
    low_r = torch.from_numpy(np.asarray(low_r)).float()
    low_r = low_r.permute(2, 0, 1)
    low_r = low_r.unsqueeze(0).cuda() / 255.0

    # model_loading
    model = Net().cuda()
    checkpoint = torch.load(config.snapshots_pth)
    model.load_state_dict(checkpoint)

    pre_l, pre_r = model(light_l, light_r, low_l, low_r)
    pre_l = pre_l.squeeze(0)
    pre_r = pre_r.squeeze(0)
    print("第", i, "张", " file_name: ", file_name)
    util.save_img(pre_l, config.save_left + "\\" + file_name)
    util.save_img(pre_r, config.save_right + "\\" + file_name.replace("left", "right"))


if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--light_l', type=str, default=r"D:\mydataset\bestbignoise\holopix\test\low_fre\low\left")
        parser.add_argument('--light_r', type=str, default=r"D:\mydataset\bestbignoise\holopix\test\low_fre\low\right")
        parser.add_argument('--low_l', type=str, default=r"D:\mydataset\bestbignoise\holopix\test\normal\low\left")
        parser.add_argument('--low_r', type=str, default=r"D:\mydataset\bestbignoise\holopix\test\normal\low\right")
        parser.add_argument('--sava_left', type=str, default=r"D:\mydataset\bestbignoise\holopix\test\normal\low\right")
        parser.add_argument('--save_right', type=str, default=r"D:\mydataset\bestbignoise\holopix\test\normal\low\right")
        parser.add_argument('--cuda', type=str, default="0")
        parser.add_argument('--snapshots_pth', type=str, default="../models/111.pth")

        config = parser.parse_args()

        file_list = os.listdir(config.light_l)
        len = len(file_list)
        i = 0
        for file_name in file_list:
            i += 1
            print(i, "/", len)
            lowlight(config, file_name, i)

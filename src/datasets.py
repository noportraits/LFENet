import glob
import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Dataset(Dataset):
    def __init__(self, light_l, light_r, low_l, low_r, gt_l, gt_r, mode, crop, random_resize=None):
        self.light_l = light_l
        self.light_r = light_r
        self.low_l = low_l
        self.low_r = low_r
        self.gt_l = gt_l
        self.gt_r = gt_r
        self.random_resize = random_resize
        self.crop = crop
        self.mode = mode
        self.paths = os.listdir(self.gt_l)

    def __getitem__(self, index):
        light_l = Image.open(self.light_l + "\\" + self.paths[index])
        light_r = Image.open(self.light_r + "\\" + self.paths[index].replace("left", "right"))
        low_l = Image.open(self.low_l + "\\" + self.paths[index])
        low_r = Image.open(self.low_r + "\\" + self.paths[index].replace("left", "right"))
        gt_l = Image.open(self.gt_l + "\\" + self.paths[index])
        gt_r = Image.open(self.gt_r + "\\" + self.paths[index].replace("left", "right"))
        x = np.concatenate((light_l, light_r, low_l, low_r, gt_l, gt_r), axis=2)

        if self.mode == 'train':
            h, w = x.shape[0], x.shape[1]
            h = random.randint(0, max(0, h - self.crop[0] - 1))
            w = random.randint(0, max(0, w - self.crop[1] - 1))
            x = x[h:h + self.crop[0], w:w + self.crop[1], :]
            if random.random() < 0.5:
                np.flip(x, axis=0)
            if random.random() < 0.5:
                np.flip(x, axis=1)
            if random.random() < 0.5:
                np.rot90(x, 2)
        if self.mode == 'val':
            h, w = x.shape[0], x.shape[1]
            h = random.randint(0, max(0, h - self.crop[0] - 1))
            w = random.randint(0, max(0, w - self.crop[1] - 1))
            x = x[h:h + self.crop[0], w:w + self.crop[1], :]
            if random.random() < 0.5:
                np.flip(x, axis=0)
            if random.random() < 0.5:
                np.flip(x, axis=1)
            if random.random() < 0.5:
                np.rot90(x, 2)

        x = self.to_tensor(x)
        light_l = x[:3, :, :]
        light_r = x[3:6, :, :]
        low_l = x[6:9, :, :]
        low_r = x[9:12, :, :]
        gt_l = x[12:15, :, :]
        gt_r = x[15:18, :, :]
        return light_l, light_r, low_l, low_r, gt_l, gt_r

    def __len__(self):
        return len(self.paths)

    def to_tensor(self, x):
        x = np.transpose(x, (2, 0, 1)) / 255.0
        x = torch.from_numpy(x).float()
        return x




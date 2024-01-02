import os
import random

import cv2
import numpy as np


def write_img(a, b, g, s_path, d_path, name):
    s_path = s_path + "\\" + name
    d_path = d_path + "\\" + name
    print("source:", s_path)
    print("out:", d_path)
    input_img = cv2.imread(s_path).astype(np.float32) / 255.0
    input_img_transformed = np.power(np.expand_dims(a * input_img, axis=(3, 4)), g[:, np.newaxis, np.newaxis])
    output_img = (b[:, np.newaxis, np.newaxis] * input_img_transformed)
    output_img_uint8 = np.squeeze(output_img)
    image = np.round(output_img_uint8 * 255.0).astype(np.uint8)
    cv2.imwrite(d_path, image)


path = "D:\\mydataset\\bestbignoise\\kitti\\test\\normal\\low\\left"
path_list = os.listdir(path)
out_path = "D:\\mydataset\\bestbignoise\\kitti\\test\\normal\\low\\left"
len = path_list.__len__()
l = 0
for file_name in path_list:
    print("***" * 100)
    print(l, "/", len)
    l += 1
    # holopix
    alpha1 = np.random.uniform(0.53, 0.58, size=3)  # 对比度
    beta1 = np.random.uniform(0.53, 0.58, size=3)  # 亮度
    gamma1 = np.random.uniform(1.3, 1.5, size=3)  # 曝光度

    alpha2 = np.random.uniform(0.8, 0.85, size=3)  # 对比度
    beta2 = np.random.uniform(0.8, 0.85, size=3)  # 亮度
    gamma2 = np.random.uniform(3, 3.2, size=3)  # 曝光度
    '''
    flickr
    alpha1 = np.random.uniform(0.62, 0.65, size=3)  # 对比度
    beta1 = np.random.uniform(0.62, 0.65, size=3)  # 亮度
    gamma1 = np.random.uniform(1.5, 1.7, size=3)  # 曝光度

    alpha2 = np.random.uniform(0.8, 0.85, size=3)  # 对比度
    beta2 = np.random.uniform(0.8, 0.85, size=3)  # 亮度
    gamma2 = np.random.uniform(3, 3.2, size=3)  # 曝光度
    '''
    '''
    kitti
    alpha1 = np.random.uniform(0.65, 0.68, size=3)  # 对比度
    beta1 = np.random.uniform(0.65, 0.68, size=3)  # 亮度
    gamma1 = np.random.uniform(1.55, 1.65, size=3)  # 曝光度

    alpha2 = np.random.uniform(0.8, 0.85, size=3)  # 对比度
    beta2 = np.random.uniform(0.8, 0.85, size=3)  # 亮度
    gamma2 = np.random.uniform(3, 3.2, size=3)  # 曝光度
    '''
    a = [[alpha1, beta1, gamma1], [alpha2, beta2, gamma2]]
    n = random.randint(1, 10)
    index = 0
    if n > 8:
        index = 1
    alpha = a[index][0]
    beta = a[index][1]
    gamma = a[index][2]
    file_left = file_name
    file_right = file_name.replace("left", "right")
    write_img(alpha, beta, gamma, path, out_path, file_left)
    write_img(alpha, beta, gamma, path.replace("left", "right"), out_path.replace("left", "right"), file_right)

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json

from transform import *


class CamVid(Dataset):
    # CamVid数据集总共有11类，可以用下面的数组进行表示，坐标表示像素值，数据对应标签
    # lables = ["sky", "building", "column pole", "road", "sidewalk", "tree", "sign symbol", "fence", "car", "pedestrian",
    #           "bicyclist"]
    # 参数为图像的根地址，和随机裁剪的尺寸
    def __init__(self, rootpth, cropsize=(640, 480),
                 randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), mode='train', *args,
                 **kwargs):
        super(CamVid, self).__init__(*args, **kwargs)
        # 确认数据集加载的类型
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.img = []
        self.label = []
        print('self.mode', self.mode)
        ## parse img directory
        # 获取图片，将图片的路径存放在{名字：路径}的字典中
        img_list_file = rootpth + "/camvid32/" + self.mode + ".txt"
        root_img_path = rootpth + "/camvid32/camvid/images/"
        root_label_path = rootpth + "/camvid32/camvid/labels/"
        with open(img_list_file, "r") as f:
            img_file_name = f.readlines()
            for data in img_file_name:
                data = data.strip()
                self.img.append(root_img_path + data)
                self.label.append(root_label_path + data.replace('.png', '_P.png'))

        self.len = len(self.img)
        print('self.len', self.mode, self.len)
        ## pre-processing
        # 定义前处理的方式
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize),
        ])

    def __getitem__(self, idx):
        # 对于dataloader加载数据集中的内容时
        impth = self.img[idx]
        lbpth = self.label[idx]
        # 图片读取
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        # 训练集进行数据增强
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        # 转换成Tensor类型
        img = self.to_tensor(img)
        # 在修改完数据类型后添加一个维度
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label

    def __len__(self):
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm

    ds = CamVid('./data', mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))
    print(len(set(uni)))

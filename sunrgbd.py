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


class SunRGBD(Dataset):
    # SunRGBD数据集总共有37+1类（1是背景），可以用下面的数组进行表示，坐标表示像素值，数据对应标签
    # 参数为图像的根地址，和随机裁剪的尺寸
    def __init__(self, rootpth, cropsize=(591, 441), mode='train', *args,
                 **kwargs):
        super(SunRGBD, self).__init__(*args, **kwargs)
        # 确认数据集加载的类型
        assert mode in ('train', 'test')
        self.mode = mode
        self.img = []
        self.label = []
        print('self.mode', self.mode)
        ## parse img directory
        # 获取图片，将图片的路径存放在{名字：路径}的字典中
        img_list_file = rootpth + "/SUN_RGBD/" + self.mode + ".txt"
        with open(img_list_file, "r") as f:
            img_file_name = f.readlines()
            for data in img_file_name:
                data = data.strip().split()
                img_path = data[0]
                label_path = data[1]
                self.img.append(img_path)
                self.label.append(label_path)

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
        if self.mode == 'train':
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
    # 参数如下：
    # dspth：数据集的路径（不带），字符串
    # cropsize：输入图像要经过缩放后输出的最终结果(默认：(591,441))，列表/元组，如[1024, 512]
    # mode：训练集类型，字符串，test表示測試集，train表示训练集，
    ds = SunRGBD('/home/disk2/ray/datasets/', mode='train')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))
    print(len(set(uni)))

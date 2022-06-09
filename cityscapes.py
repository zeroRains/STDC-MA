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



class CityScapes(Dataset):
    # 参数为图像的根地址，和随机裁剪的尺寸
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', 
    randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(CityScapes, self).__init__(*args, **kwargs)
        # 确认数据集加载的类型
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255
        #     获取标签信息，因为原数据集有30多类，现在要换成19类，因此需要一个映射的字典
        with open('./cityscapes_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {el['id']: el['trainId'] for el in labels_info}
        

        ## parse img directory
        # 获取图片，将图片的路径存放在{名字：路径}的字典中
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            # 将单个键值对加入到字典中
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        # 获取标签和上面的操作差不多
        self.labels = {}
        gtnames = []
        gtpth = osp.join(rootpth, 'gtFine', mode)
        folders = os.listdir(gtpth)
        for fd in folders:
            # if fd == "strasbourg":
            #     continue
            fdpth = osp.join(gtpth, fd)
            lbnames = os.listdir(fdpth)
            lbnames = [el for el in lbnames if 'labelIds' in el]
            names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
            lbpths = [osp.join(fdpth, el) for el in lbnames]
            gtnames.extend(names)
            self.labels.update(dict(zip(names, lbpths)))
        #
        self.imnames = imgnames
        self.len = len(self.imnames)
        print('self.len', self.mode, self.len)
        # print(set(imgnames))
        # print(set(gtnames))
        assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        # 定义前处理的方式
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomScale(randomscale),
            # RandomScale((0.125, 1)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0)),
            # RandomScale((0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        # 对于dataloader加载数据集中的内容时
        # 先获取图像的名字，以此以键搜到对应的值
        fn  = self.imnames[idx]
        impth = self.imgs[fn]
        lbpth = self.labels[fn]
        # 图片读取
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        # 训练集进行数据增强
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        # 转换成Tensor类型
        img = self.to_tensor(img)
        # 在修改完数据类型后添加一个维度
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        # 通过遍历字典将对应的像素(标签,id)转化为训练的像素(训练Id,训练标签)
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    from tqdm import tqdm
    # 训练集
    # 参数如下：
    # dspth：数据集的路径（不带），字符串
    # cropsize：输入图像要经过缩放后输出的最终结果(默认：(640, 480))，列表/元组，如[1024, 512]
    # mode：训练集类型，字符串，val表示验证集，train表示训练集，
    # randomscale：这个是随机缩放的比例，列表/元组，如：(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5)
    ds = CityScapes('/home/disk2/ray/datasets/cityscapes/',  mode='val')
    # 验证集（测试集好像是不公开的，用验证集就行）
    # ds = CityScapes('./data/', mode='val')
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))


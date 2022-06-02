import json

from models.model_stages_msc_fapn import BiSeNet, FeatureSelectionModule, FeatureAlign_V2, ConvBNReLU
from cityscapes import CityScapes
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import cv2
import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

cuda_id = 3
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(cuda_id)


# function for colorizing a label image:
def visual(module, inputs, outputs):
    if isinstance(module, FeatureSelectionModule):
        x = inputs[0][0]
        y = outputs[0]
        for i in range(10):
            plt.imshow(x[i].detach().cpu().numpy())
            plt.savefig(f"visiable/featuremap/1low_level_input{i}.jpg")
            plt.imshow(y[i].detach().cpu().numpy())
            plt.savefig(f"visiable/featuremap/1low_level_output{i}.jpg")
    if isinstance(module, FeatureAlign_V2):
        # high_level = inputs[1][0].detach().cpu().numpy()
        # index = np.unravel_index(high_level.argmax(), high_level.shape)
        # print(index)
        y = outputs[0]
        for i in range(10):
            plt.imshow(y[i].detach().cpu().numpy())
            plt.savefig(f"visiable/featuremap/1fam_output{i}.jpg")
    if isinstance(module, ConvBNReLU):
        high_level = outputs[0].detach().cpu().numpy()
        for i in range(10):
            plt.imshow(2 * high_level[i + 40])
            plt.savefig(f"visiable/featuremap/1high_level_input{i}.jpg")
    exit()


def draw_color(img):
    label_to_color = {
        0: [128, 64, 128],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60],
        12: [255, 0, 0],
        13: [0, 0, 142],
        14: [0, 0, 70],
        15: [0, 60, 100],
        16: [0, 80, 100],
        17: [0, 0, 230],
        18: [119, 11, 32],
        19: [81, 0, 81],
        255: [0, 0, 0]
    }

    img_height, img_width = img.shape

    img_color = np.zeros((img_height, img_width, 3))
    for row in range(img_height):
        for col in range(img_width):
            label = img[row, col]
            img_color[row, col] = np.array(label_to_color[label])
    return img_color


class Predictor(object):
    def __init__(self, scale=0.5, ignore_label=255):
        self.ignore_label = ignore_label
        self.scale = scale

    def __call__(self, net, imgs, n_classes, trick=None, patch=None, mask=None, scales=None):

        N, _, H, W = imgs.shape

        size = imgs.size()[-2:]

        imgs = imgs.cuda(cuda_id)
        if not isinstance(trick, type(None)) \
                and not isinstance(patch, type(None)) \
                and not isinstance(mask, type(None)):
            imgs = imgs.cuda(3)
            imgs = trick(imgs, patch, mask)

        N, C, H, W = imgs.size()
        new_hw = [int(H * self.scale), int(W * self.scale)]

        imgs = F.interpolate(imgs, new_hw, mode='bilinear', align_corners=True)

        # 这个结果只取了两个ARM处理，并经过FFM融合的结果
        if scales is not None:
            logits = net(imgs, scales=scales)[0]
        else:
            logits = net(imgs)[0]

        # print("zhe")
        # print(imgs.device)
        logits = F.interpolate(logits, size=size,
                               mode='bilinear', align_corners=True)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds


def convert_labels(lb_map, label):
    # 通过遍历字典将对应的像素(标签,id)转化为训练的像素(训练Id,训练标签)
    for k, v in lb_map.items():
        label[label == k] = v
    return label


def predcited(model_path, output_path):
    image_path = "./visiable/images"
    images = os.listdir(image_path)
    # dsval = CityScapes(image_path, mode='val')
    # dl = DataLoader(dsval,
    #                 batch_size=1,
    #                 shuffle=False,
    #                 num_workers=0,
    #                 drop_last=False)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    n_classes = 19
    # net = BiSeNet(backbone="STDCNet1446", n_classes=19, pretrain_model=None, use_boundary_2=False,
    #               use_boundary_4=False, use_boundary_8=True, use_boundary_16=False, use_conv_last=False)
    # net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net = torch.load(model_path, map_location='cpu')
    for name, m in net.named_modules():
        # if isinstance(m, FeatureAlign_V2):
        #     m.register_forward_hook(visual)
        # if isinstance(m, FeatureSelectionModule):
        #     m.register_forward_hook(visual)
        if isinstance(m, ConvBNReLU):
            m.register_forward_hook(visual)
    # net = BiSeNet(backbone="STDCNet1446", n_classes=19, pretrain_model=None, use_boundary_2=False, use_boundary_4=False,
    #               use_boundary_8=True, use_boundary_16=False, use_conv_last=False)
    # net.load_state_dict(parmas)
    net.eval()
    net.cuda(cuda_id)
    with torch.no_grad():
        for path in images:
            img = Image.open(os.path.join(image_path, path)).convert('RGB')
            image = np.array(img)
            img = to_tensor(img)
            img = img.unsqueeze(dim=0)
            scales = [0.25, 0.5, 1.0, 1.5, 2.0]
            pre = Predictor()
            res50 = pre(net, img, n_classes, scales=scales).squeeze(dim=0)
            pre = Predictor(scale=0.75)
            res75 = pre(net, img, n_classes, scales=scales).squeeze(dim=0)
            res50 = res50.cpu().numpy()
            res75 = res75.cpu().numpy()
            color50 = draw_color(res50)
            color75 = draw_color(res75)
            cv2.imwrite(os.path.join(output_path, "r50", path), color50)
            cv2.imwrite(os.path.join(output_path, "r75", path), color75)
            print(f"{path} work finished!")


if __name__ == '__main__':
    # im_path = "visiable/gt"
    # images = os.listdir(im_path)
    # with open('./cityscapes_info.json', 'r') as fr:
    #     labels_info = json.load(fr)
    # lb_map = {el['id']: el['trainId'] for el in labels_info}
    # for im in images:
    #     img = cv2.imread(os.path.join(im_path, im), -1)
    #     img = convert_labels(lb_map, img)
    #     # img[img == 255] = 0
    #     # print(img.shape)
    #     color = draw_color(img)
    #     cv2.imwrite(os.path.join(im_path, im), color)
    #     print(f"{im} finished!")
    model_path = "./checkpoints/MSC_FaPN_RMI_optim_STDC2-Seg/pths/model_maxmIOU50.pth"
    output_path = "./visiable/msc"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.mkdir(os.path.join(output_path, 'r50'))
        os.mkdir(os.path.join(output_path, 'r75'))
    print(f"load_model：{model_path}")
    print(f"output_path：{output_path}")
    predcited(model_path, output_path)

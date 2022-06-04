#!/usr/bin/python
# -*- encoding: utf-8 -*-
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from nets.stdcnet import STDCNet1446, STDCNet813
# from modules.bn import InPlaceABNSync as BatchNorm2d
from torch.nn import BatchNorm2d


def make_attn_head(in_ch, out_ch):
    # 256
    bot_ch = 256
    # 创建有序字典，3x3的same卷积
    od = OrderedDict([('conv0', nn.Conv2d(in_ch, bot_ch, kernel_size=3,
                                          padding=1, bias=False)),
                      ('bn0', nn.BatchNorm2d(bot_ch)),
                      ('re0', nn.ReLU(inplace=True))])
    # True
    # 3x3的Same卷积
    if True:
        od['conv1'] = nn.Conv2d(bot_ch, bot_ch, kernel_size=3, padding=1,
                                bias=False)
        od['bn1'] = nn.BatchNorm2d(bot_ch)
        od['re1'] = nn.ReLU(inplace=True)

    # 1x1卷积
    od['conv2'] = nn.Conv2d(bot_ch, out_ch, kernel_size=1, bias=False)
    od['sig'] = nn.Sigmoid()
    # 有序字典装进Sequential里
    attn_head = nn.Sequential(od)
    # init_attn(attn_head)
    return attn_head


def ResizeX(x, scale_factor):
    x_scaled = torch.nn.functional.interpolate(
        x, scale_factor=scale_factor, mode='bilinear',
        align_corners=False, recompute_scale_factor=True)
    return x_scaled


def fmt_scale(prefix, scale):
    """
    format scale name

    :prefix: a string that is the beginning of the field name
    :scale: a scale value (0.25, 0.5, 1.0, 2.0)
    """

    scale_str = str(float(scale))
    scale_str.replace('.', '')
    return f'{prefix}_{scale_str}x'


def scale_as(x, y):
    y_size = y.size(2), y.size(3)
    x_scaled = torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear',
        align_corners=False)
    return x_scaled


# BatchNorm2d = nn.BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, with_attention=False, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        # self.gate = GCT(in_chan)
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        # self.bn = BatchNorm2d(out_chan)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.with_attention = with_attention

    def forward(self, x):
        # if self.with_attention:
        # x = self.gate(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        # self.bn_atten = BatchNorm2d(out_chan, activation='none')

        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        # 对输入的图像先进行一个卷积操作(same类型的)
        feat = self.conv(x)
        # 然后做一个全局平均池化(这个使用了整个图大小的卷积核)，最后会变成nxc维度的值
        # 权重就是注意力的本质了
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        # 然后乘上就可以了
        out = torch.mul(feat, atten)
        return out


## BisNet的上下文路径
class ContextPath(nn.Module):
    def __init__(self, backbone='CatNetSmall', pretrain_model='', use_conv_last=False, *args, **kwargs):
        super(ContextPath, self).__init__()
        # 设置主干网络的名字
        self.backbone_name = backbone
        if backbone == 'STDCNet1446':
            # STDC网络(感觉有点出入，这个模块好像只用了一个FC层)输出有5个，分别表示Stage1,2,3,4,5的输出结果
            self.backbone = STDCNet1446(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            # ARM模块，注意力细化模块，原图乘上注意力，输出为Nx128xHxW
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)

        elif backbone == 'STDCNet813':
            self.backbone = STDCNet813(pretrain_model=pretrain_model, use_conv_last=use_conv_last)
            self.arm16 = AttentionRefinementModule(512, 128)
            inplanes = 1024
            if use_conv_last:
                inplanes = 1024
            self.arm32 = AttentionRefinementModule(inplanes, 128)
            self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
            self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)
        else:
            print("backbone is not in backbone lists")
            exit(0)

    def forward(self, x):
        # 获取宽高
        H0, W0 = x.size()[2:]
        # 获取Stage1,2,3,4,5的输出
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)
        # 记录3,4,5的宽高
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        # 使用全局平均池化，这个的结果维度应该为NxC
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        # 然后又做一个上采样样恢复成原图尺寸？
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')
        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)
        # 第二个ARM模块(结构图从上往下看)
        feat32_arm = self.arm32(feat32)
        # 将ARM的结果与avg_up进行加和
        feat32_sum = feat32_arm + avg_up
        # 然后进行上采样到和stage4觉得结果一样的大小
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        # 做一个1x1卷积
        feat32_up = self.conv_head32(feat32_up)

        # 第一个ARM
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        # 上采样至stage3的尺寸
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat2, feat4, feat8, feat16, feat16_up, feat32_up  # x8, x16


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        # 全都采用1x1卷积的方式进行融合
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        # 对输入的特征进行拼接
        fcat = torch.cat([fsp, fcp], dim=1)
        # 采用1x1卷积的方式先做一次特征提取
        feat = self.convblk(fcat)
        # 使用平均池化的方式创建注意力
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        # 将特征图与注意力进行相乘
        feat_atten = torch.mul(feat, atten)
        # 将原图特征与注意力处理的特征图结果进行相加
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(nn.Module):
    def __init__(self, backbone, n_classes, pretrain_model='', use_boundary_2=False, use_boundary_4=False,
                 use_boundary_8=False, use_boundary_16=False, use_conv_last=False, heat_map=False, *args, **kwargs):
        super(BiSeNet, self).__init__()

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8
        self.use_boundary_16 = use_boundary_16
        # self.heat_map = heat_map
        # 上下文路径，输出结果包括stage1,2,3,4的结果，和stage4,5经过ARM模块处理后的结果，一共5个结果
        self.cp = ContextPath(backbone, pretrain_model, use_conv_last=use_conv_last)
        # 这里的STDC表示的STDC网络，1446编号的是STDC2，813编号的是STDC1，两者区别在于4，5层重复使用STDC基本模块单元的次数不同
        # 我寻思着这两个参数也没有什么不同吧
        if backbone == 'STDCNet1446':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        elif backbone == 'STDCNet813':
            conv_out_inplanes = 128
            sp2_inplanes = 32
            sp4_inplanes = 64
            sp8_inplanes = 256
            sp16_inplanes = 512
            inplane = sp8_inplanes + conv_out_inplanes

        else:
            print("backbone is not in backbone lists")
            exit(0)
        # 特征融合模块
        self.ffm = FeatureFusionModule(inplane, 256)
        # 这个就是对输入的特征图，将其通道数转化成类的数量
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, n_classes)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)

        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)

        self.scale_attn = make_attn_head(in_ch=256, out_ch=1)

    def _fwd(self, x):
        H, W = x.size()[2:]
        # 这个是空间上下文路径的计算，主要是Stage1,2,3,4的特征图和4,5经过ARM模块处理后的特征图
        feat_res2, feat_res4, feat_res8, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        # Stage1做图像分割的mask
        feat_out_sp2 = self.conv_out_sp2(feat_res2)
        # stage2做图像分割的mask
        feat_out_sp4 = self.conv_out_sp4(feat_res4)
        # stage3做图像分割的mask
        feat_out_sp8 = self.conv_out_sp8(feat_res8)
        # stage4做图像分割的mask
        feat_out_sp16 = self.conv_out_sp16(feat_res16)

        # 做图像特征融合，将stage3和stage3经过ARM处理的特征图进行融合
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        # 输出最终的mask
        feat_out = self.conv_out(feat_fuse)
        # 这个是经过第一个ARM处理过的图像进行图像分割产书馆不通宵生mask
        feat_out16 = self.conv_out16(feat_cp8)
        # 这个是经过第二个ARM处理过的图像进行图像分割产生的mask
        feat_out32 = self.conv_out32(feat_cp16)

        # 这个是最终的mask上采样到原图大小
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        # 这个是经过第一个ARM处理的图像上采样到原图大小
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        # 这个是经过第二个ARM处理的图像上采样到原图大小
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        # 这里就是注意力时间了，这里输入的是ocr(两个3x3卷积，一个1x1卷积，维度没有改变)，最后一层用sigmoid
        attn = self.scale_attn(feat_fuse)

        attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=True)
        # 字典形式输出
        return feat_out, feat_out16, feat_out32, feat_out_sp8, attn

    def two_scale_forward(self, inputs):
        """
        Do we supervised both aux outputs, lo and high scale?
        Should attention be used to combine the aux output?
        Normally we only supervise the combined 1x output

        If we use attention to combine the aux outputs, then
        we can use normal weighting for aux vs. cls outputs
        """
        x_1x = inputs

        # 缩放到0.5x
        x_lo = ResizeX(x_1x, 0.5)

        feat_out, _, _, _, attn = self._fwd(x_lo)
        pred_05x = feat_out
        p_lo = pred_05x
        logit_attn = attn
        attn_05x = logit_attn

        # 这个使用的是1.0x的
        feat_out1, feat_out16, feat_out32, feat_out_sp8, attn1 = self._fwd(x_1x)
        pred_10x = feat_out1
        p_1x = pred_10x

        # 注意力与0.5x的cls进行像素级相乘
        p_lo = logit_attn * p_lo
        # soft object representation 乘上注意力
        # 将0.5x乘上注意力的结果进行一个上采样至和1.0x一样大
        p_lo = scale_as(p_lo, p_1x)

        # 注意力也这么做
        logit_attn = scale_as(logit_attn, p_1x)

        # combine lo and hi predictions with attention
        # 然后把低尺度没有关注到的内容，从高尺度中进行提取
        joint_pred = p_lo + (1 - logit_attn) * p_1x
        return joint_pred, feat_out16, feat_out32, feat_out_sp8

    def nscale_forward(self, inputs, scales):
        x_1x = inputs

        assert 1.0 in scales, 'expected 1.0 to be the target scale'
        # Lower resolution provides attention for higher rez predictions,
        # so we evaluate in order: high to low
        # 从大到小对尺度进行排序
        scales = sorted(scales, reverse=True)

        pred = None
        feat_out16 = feat_out32 = feat_out_sp8 = None
        output_dict = {}

        for s in scales:
            x = ResizeX(x_1x, s)
            # 网络正经推理
            cls_out, feat_out16p, feat_out32p, feat_out_sp8p, attn_out = self._fwd(x)
            output_dict[fmt_scale('pred', s)] = cls_out
            if s != 2.0:
                output_dict[fmt_scale('attn', s)] = attn_out

            if pred is None:
                pred = cls_out
            if s == 1.0:
                feat_out16, feat_out32, feat_out_sp8 = feat_out16p, feat_out32p, feat_out_sp8p

            elif s >= 1.0:
                # downscale previous
                pred = scale_as(pred, cls_out)
                pred = attn_out * cls_out + (1 - attn_out) * pred

            else:
                # s < 1.0: upscale current
                cls_out = attn_out * cls_out
                cls_out = scale_as(cls_out, pred)
                attn_out = scale_as(attn_out, pred)

                pred = cls_out + (1 - attn_out) * pred
        return pred, feat_out16, feat_out32, feat_out_sp8

    def forward(self, x, scales=None):
        if scales and not self.training:
            return self.nscale_forward(x, scales)
        return self.two_scale_forward(x)


if __name__ == "__main__":
    net = BiSeNet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')

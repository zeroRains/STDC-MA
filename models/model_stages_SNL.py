#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from nets.stdcnet_snl import STDCNet1446, STDCNet813
# from modules.bn import InPlaceABNSync as BatchNorm2d
from torch.nn import BatchNorm2d


class SNLStage(nn.Module):
    def __init__(self, inplanes, planes, stage_num=1, use_scale=False, relu=False, aff_kernel='dot'):
        super(SNLStage, self).__init__()
        self.down_channel = planes
        self.output_channel = inplanes
        self.num_block = stage_num
        self.input_channel = inplanes
        # self.softmax = nn.Softmax(dim=2)
        self.aff_kernel = aff_kernel
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.use_scale = use_scale

        layers = []
        for i in range(stage_num):
            layers.append(SNLUnit(inplanes, planes, relu=relu))

        self.stages = nn.Sequential(*layers)

    def DotKernel(self, x):

        t = self.t(x)
        p = self.p(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)

        ####
        att = torch.bmm(torch.relu(t), torch.relu(p))
        att += att.permute(0, 2, 1)
        att = att / 2

        d = torch.sum(att, dim=2)
        d[d != 0] = torch.sqrt(1.0 / d[d != 0])
        att *= d.unsqueeze(1)
        att *= d.unsqueeze(2)
        ####

        return att

    def forward(self, x):

        if self.aff_kernel == 'dot':
            att = self.DotKernel(x)
        # elif self.aff_kernel == 'embedgassian':
        #    att = self.EbdedGassKernel(x)
        # elif self.aff_kernel == "gassian":
        #    att = self.GassKernel(x)
        # elif self.aff_kernel == 'rbf':
        #    att = self.RBFGassKeneral(x)
        else:
            exit(0)

        out = x

        for cur_stage in self.stages:
            out = cur_stage(out, att)

        return out


class SNLUnit(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, relu=False, aff_kernel='dot'):
        self.use_scale = use_scale

        super(SNLUnit, self).__init__()

        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.w_1 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)
        self.w_2 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, bias=False)

        self.is_relu = relu
        if self.is_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, att):
        residual = x

        g = self.g(x)

        b, c, h, w = g.size()

        g = g.view(b, c, -1).permute(0, 2, 1)

        x_1 = g.permute(0, 2, 1).contiguous().view(b, c, h, w)
        x_1 = self.w_1(x_1)

        out = x_1

        x_2 = torch.bmm(att, g)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = x_2.contiguous()
        x_2 = x_2.view(b, c, h, w)
        x_2 = self.w_2(x_2)
        out = out - x_2

        out = self.bn(out)

        if self.is_relu:
            out = self.relu(out)

        out = out + residual

        return out


# BatchNorm2d = nn.BatchNorm2d

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, with_attention=False, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
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
        feat2, feat4, feat8, snl, feat16, feat32 = self.backbone(x)
        # 记录3,4,5的宽高
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        # 使用全局平均池化，这个的结果维度应该为NxC
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        # 然后又做一个上采样样恢复成原图尺寸？
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

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

        return feat2, feat4, feat8, snl,feat16, feat16_up, feat32_up  # x8, x16


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

        # SNL
        # self.snl = SNLStage(sp8_inplanes, sp8_inplanes * 2)

    def forward(self, x):
        H, W = x.size()[2:]
        # 这个是空间上下文路径的计算，主要是Stage1,2,3,4的特征图和4,5经过ARM模块处理后的特征图
        feat_res2, feat_res4, feat_res8, snl, feat_res16, feat_cp8, feat_cp16 = self.cp(x)
        # Stage1做图像分割的mask
        feat_out_sp2 = self.conv_out_sp2(feat_res2)
        # stage2做图像分割的mask
        feat_out_sp4 = self.conv_out_sp4(feat_res4)
        # stage3做图像分割的mask
        feat_out_sp8 = self.conv_out_sp8(feat_res8)
        # stage4做图像分割的mask
        feat_out_sp16 = self.conv_out_sp16(feat_res16)
        # 统一Nonlocal块
        # feat_res8 = self.snl(feat_res8)

        # 做图像特征融合，将stage3和stage3经过ARM处理的特征图进行融合
        feat_fuse = self.ffm(snl, feat_cp8)
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

        # 1，2，3都用
        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8

        # 2,3都用
        if (not self.use_boundary_2) and self.use_boundary_4 and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp4, feat_out_sp8
        # 只要3
        if (not self.use_boundary_2) and (not self.use_boundary_4) and self.use_boundary_8:
            return feat_out, feat_out16, feat_out32, feat_out_sp8
        # 除了输出结果其他都不要
        if (not self.use_boundary_2) and (not self.use_boundary_4) and (not self.use_boundary_8):
            return feat_out, feat_out16, feat_out32


if __name__ == "__main__":
    net = BiSeNet('STDCNet813', 19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 768, 1536).cuda()
    out, out16, out32 = net(in_ten)
    print(out.shape)
    torch.save(net.state_dict(), 'STDCNet813.pth')

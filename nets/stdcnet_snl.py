import torch
import torch.nn as nn
from torch.nn import init
import math



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


class ConvX(nn.Module):
    # conv + BN + Relu
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, with_attention=False):
        super(ConvX, self).__init__()
        # self.gate = GCT(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.with_attention = with_attention

    def forward(self, x):
        # if self.with_attention:
        # x = self.gate(x)
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


# 我感觉这个就是STDC模块
class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1
        # 根据层数缩减输出的通道数，这个是STDC计算单元的构造
        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        # print(out.shape)
        # exit(0)
        return out


# STDC2Net
class STDCNet1446(nn.Module):
    def __init__(self, base=64, layers=[4, 5, 3], block_num=4, type="car", num_classes=1000, dropout=0.20,
                 pretrain_model='', use_conv_last=False):
        super(STDCNet1446, self).__init__()
        # 将STDC模块输出的特征图进行拼接
        if type == "car":
            block = CatBottleneck
        #     这个是使用原输入经过变换与输出的特征图拼接结果进行相加
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        # 构建Stage3,4,5
        self.features = self._make_layers(base, layers, block_num, block)
        # ConvX6
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        # FC层
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        # 神经失活
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        # 分别表示Stage1,2,3,4,5的层
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.snl = SNLStage(256, 256)
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

    def _make_layers(self, base, layers, block_num, block):
        features = []
        # 将输入的图像先拓展到base的通道数
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]
        # 根据参数列表设计对应个数的STDC模块
        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        snl = self.snl(feat8)
        feat16 = self.x16(snl)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, snl ,feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


# STDC1Net
class STDCNet813(nn.Module):
    def __init__(self, base=64, layers=[2, 2, 2], block_num=4, type="car", num_classes=1000, dropout=0.20,
                 pretrain_model='', use_conv_last=False):
        super(STDCNet813, self).__init__()
        if type == "car":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, block)
        self.conv_last = ConvX(base * 16, max(1024, base * 16), 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(max(1024, base * 16), max(1024, base * 16), bias=False)
        self.bn = nn.BatchNorm1d(max(1024, base * 16))
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(max(1024, base * 16), num_classes, bias=False)

        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.snl = SNLStage(256, 256)
        self.x16 = nn.Sequential(self.features[4:6])
        self.x32 = nn.Sequential(self.features[6:])

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(3, base // 2, 3, 2)]
        features += [ConvX(base // 2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base * 4, block_num, 2))
                elif j == 0:
                    features.append(block(base * int(math.pow(2, i + 1)), base * int(math.pow(2, i + 2)), block_num, 2))
                else:
                    features.append(block(base * int(math.pow(2, i + 2)), base * int(math.pow(2, i + 2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        snl = self.snl(feat8)
        feat16 = self.x16(snl)
        feat32 = self.x32(feat16)
        if self.use_conv_last:
            feat32 = self.conv_last(feat32)

        return feat2, feat4, feat8, snl, feat16, feat32

    def forward_impl(self, x):
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        # out = self.bn(out)
        out = self.relu(out)
        # out = self.relu(self.bn(self.fc(out)))
        out = self.dropout(out)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    model = STDCNet813(num_classes=1000, dropout=0.00, block_num=4)
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    torch.save(model.state_dict(), 'car.pth')
    print(y.size())

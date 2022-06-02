#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision.utils

from logger import setup_logger

from models.model_stages import BiSeNet

from cityscapes import CityScapes
from loss.loss import OhemCELoss
from loss.detail_loss import DetailAggregateLoss
from evaluation import MscEvalV0
from optimizer_loss import Optimizer
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
# import torch.distributed as dist
from PIL import Image
import os
import os.path as osp
import logging
import time
import datetime
import argparse

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger()
CUDA_ID = 3
torch.cuda.set_device(CUDA_ID)
# summary = SummaryWriter("logs/board"),
patch_size = 40


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


# --pretrain_path
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--local_rank',
        dest='local_rank',
        type=int,
        default=-1,
    )
    parse.add_argument(
        '--n_workers_train',
        dest='n_workers_train',
        type=int,
        default=12,
    )
    parse.add_argument(
        '--n_workers_val',
        dest='n_workers_val',
        type=int,
        default=1,
    )
    # batch size
    parse.add_argument(
        '--n_img_per_gpu',
        dest='n_img_per_gpu',
        type=int,
        default=8,
    )
    # 最大迭代次数
    parse.add_argument(
        '--max_iter',
        dest='max_iter',
        type=int,
        default=150000,
    )
    # 保存频率
    parse.add_argument(
        '--save_iter_sep',
        dest='save_iter_sep',
        type=int,
        default=1000,
    )
    parse.add_argument(
        '--warmup_steps',
        dest='warmup_steps',
        type=int,
        default=500,
    )
    # 运行模式
    parse.add_argument(
        '--mode',
        dest='mode',
        type=str,
        default='train',
    )
    parse.add_argument(
        '--ckpt',
        dest='ckpt',
        type=str,
        default="./checkpoints/STDC2-Seg/model_maxmIOU75.pth",
    )
    # 模型路径
    parse.add_argument(
        '--respath',
        dest='respath',
        type=str,
        default="checkpoints/with_update_STDC2-Seg/",
    )
    # 主干网络
    parse.add_argument(
        '--backbone',
        dest='backbone',
        type=str,
        default='STDCNet1446',
    )
    # 预训练模型
    parse.add_argument(
        '--pretrain_pathpretrain_path',
        dest='pretrain_path',
        type=str,
        default=None,
    )
    parse.add_argument(
        '--use_conv_last',
        dest='use_conv_last',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_2',
        dest='use_boundary_2',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_4',
        dest='use_boundary_4',
        type=str2bool,
        default=False,
    )
    parse.add_argument(
        '--use_boundary_8',
        dest='use_boundary_8',
        type=str2bool,
        default=True,
    )
    parse.add_argument(
        '--use_boundary_16',
        dest='use_boundary_16',
        type=str2bool,
        default=False,
    )
    return parse.parse_args()


class GenerateImage(nn.Module):
    def __init__(self):
        super(GenerateImage, self).__init__()
        # self.conv = nn.Conv2d(3, 3, 1, 1)
        # self.STDC = nn.Sequential(model)

    def forward(self, img, patch, mask):
        ih, iw = img.shape[2:]
        ib = img.shape[0]
        # print(img.device)
        img_batch = img.clone()
        # print(img_batch.device)
        # print(patch[0].shape)
        # exit(0)
        # 左，右，上，下
        pad_way = [[0, iw - patch_size, 0, ih - patch_size],
                   [iw - patch_size, 0, 0, ih - patch_size],
                   [0, iw - patch_size, ih - patch_size, 0],
                   [iw - patch_size, 0, ih - patch_size, 0]]
        mask = mask.unsqueeze(dim=0).repeat([ib, 1, 1, 1])
        # make mask
        for index, way in enumerate(pad_way):
            mask_temp = F.pad(mask, (way[0], way[1], way[2], way[3]))
            patch_temp = patch[index].unsqueeze(dim=0).repeat([ib, 1, 1, 1])
            patch_temp = F.pad(patch_temp, (way[0], way[1], way[2], way[3]))
            mask_temp = mask_temp.type(torch.float32)
            img_batch = torch.where(mask_temp > 0.00001, patch_temp, img_batch)
        return img_batch


def generate_patch(load_from_file=None, is_random=False, is_cmyk=0):
    xpatch_size, ypatch_size = patch_size, patch_size
    # load a image from local patch
    if load_from_file is not None:
        patch = Image.open(load_from_file)
        patch = patch.resize((xpatch_size, ypatch_size))
        patch = transforms.PILToTensor()(patch) / 255.
        return patch
    if is_random:
        if is_cmyk:
            return torch.rand((4, xpatch_size, ypatch_size))
        else:
            return torch.rand((3, xpatch_size, ypatch_size))
    if is_cmyk:
        return torch.full((4, xpatch_size, ypatch_size), 0.5)
    else:
        return torch.full((3, xpatch_size, ypatch_size), 0.5)


def train():
    args = parse_args()
    # 设置模型保存路径
    save_pth_path = os.path.join(args.respath, 'pths')
    dspth = './data'

    print(save_pth_path)
    print(osp.exists(save_pth_path))
    # if not osp.exists(save_pth_path) and dist.get_rank()==0:
    if not osp.exists(save_pth_path):
        os.makedirs(save_pth_path)

    # torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(
    #             backend = 'nccl',
    #             init_method = 'tcp://127.0.0.1:8081',
    #             world_size = torch.cuda.device_count(),
    #             rank=args.local_rank
    #             )
    # print("你行不行")
    # 设置日志文件
    setup_logger(args.respath)
    ## dataset
    n_classes = 19
    n_img_per_gpu = args.n_img_per_gpu
    n_workers_train = args.n_workers_train
    n_workers_val = args.n_workers_val
    use_boundary_16 = args.use_boundary_16
    use_boundary_8 = args.use_boundary_8
    use_boundary_4 = args.use_boundary_4
    use_boundary_2 = args.use_boundary_2

    mode = args.mode
    # 输出尺寸
    cropsize = [1024, 512]
    # 随机缩放
    randomscale = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5)

    # if dist.get_rank()==0:
    #     logger.info('n_workers_train: {}'.format(n_workers_train))
    #     logger.info('n_workers_val: {}'.format(n_workers_val))
    #     logger.info('use_boundary_2: {}'.format(use_boundary_2))
    #     logger.info('use_boundary_4: {}'.format(use_boundary_4))
    #     logger.info('use_boundary_8: {}'.format(use_boundary_8))
    #     logger.info('use_boundary_16: {}'.format(use_boundary_16))
    #     logger.info('mode: {}'.format(args.mode))
    # 加载数据集
    ds = CityScapes(dspth, cropsize=cropsize, mode=mode, randomscale=randomscale)
    # sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=False,
                    num_workers=n_workers_train,

                    pin_memory=False,
                    drop_last=True)
    # exit(0)
    dsval = CityScapes(dspth, mode='val', randomscale=randomscale)
    # sampler_val = torch.utils.data.distributed.DistributedSampler(dsval)
    dlval = DataLoader(dsval,
                       batch_size=2,
                       shuffle=False,
                       num_workers=n_workers_val,
                       drop_last=False)

    ## model
    ignore_idx = 255
    # net = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path,
    #               use_boundary_2=use_boundary_2, use_boundary_4=use_boundary_4, use_boundary_8=use_boundary_8,
    #               use_boundary_16=use_boundary_16, use_conv_last=args.use_conv_last)
    net = torch.load("./checkpoints/edge_optim_STDC2-Seg/pths/model_maxmIOU75.pth")
    net.cuda(CUDA_ID)
    # net.train()
    # net = nn.parallel.DistributedDataParallel(net,
    #                                           device_ids=[args.local_rank, ],
    #                                           output_device=args.local_rank,
    #                                           find_unused_parameters=True
    #                                           )

    score_thres = 0.7
    # 这个算的什么我也看不明白，batch*h*w//16
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    criteria_p = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_16 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    criteria_32 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    # 细节损失
    boundary_loss_func = DetailAggregateLoss()
    ## optimizer
    maxmIOU50 = 0.
    maxmIOU75 = 0.
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = args.max_iter
    save_iter_sep = args.save_iter_sep
    power = 0.9
    warmup_steps = args.warmup_steps
    warmup_start_lr = 1e-5
    eval_times = 0

    # if dist.get_rank()==0:
    #     print('max_iter: ', max_iter)
    #     print('save_iter_sep: ', save_iter_sep)
    #     print('warmup_steps: ', warmup_steps)
    # optim = Optimizer(
    #     model=net,
    #     loss=boundary_loss_func,
    #     lr0=lr_start,
    #     momentum=momentum,
    #     wd=weight_decay,
    #     warmup_steps=warmup_steps,
    #     warmup_start_lr=warmup_start_lr,
    #     max_iter=max_iter,
    #     power=power)
    # 生成check图像，并且冻结整个网络的参数
    for params in net.parameters():
        params.requires_grad = False

    patches = []
    for i in range(4):
        adv_patch_cpu = generate_patch(is_random=True, is_cmyk=0)
        adv_patch = adv_patch_cpu.cuda(CUDA_ID)
        adv_patch.requires_grad_(True)
        patches.append(adv_patch)
    # adv_patch_cpu = generate_patch(is_random=True, is_cmyk=0)
    # # adv_patch_cpu.requires_grad_(True)
    # adv_patch = adv_patch_cpu.cuda(CUDA_ID)
    # adv_patch.requires_grad_(True)
    # adv_patch_cpu.requires_grad = True
    mask_patch = torch.ones_like(patches[0])
    trick = GenerateImage().cuda(CUDA_ID)

    # trick.train()
    optim = torch.optim.Adam(patches, lr=0.0001, betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0)
    ## train loop
    msg_iter = 50
    loss_avg = []
    loss_boundery_bce = []
    loss_boundery_dice = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            # 遍历数据集
            im, lb = next(diter)
            if not im.size()[0] == n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            # sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)

        # print(im.shape)
        # exit(0)
        #     加入到cuda中

        mask_patch = mask_patch.cuda(CUDA_ID)
        im = im.cuda(CUDA_ID)

        # im = trick(im, adv_patch_cpu, mask_patch)
        lb = lb.cuda(CUDA_ID)
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)
        # test_data = net(im)
        # print(test_data)
        # print(type(test_data))
        # print(len(test_data))
        # exit(0)
        # out, out16, out32, detail8 = net(im, adv_patch_cpu, mask_patch)
        if use_boundary_2 and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail2, detail4, detail8 = net(trick(im, patches, mask_patch))

        if (not use_boundary_2) and use_boundary_4 and use_boundary_8:
            out, out16, out32, detail4, detail8 = net(trick(im, patches, mask_patch))

        if (not use_boundary_2) and (not use_boundary_4) and use_boundary_8:
            out, out16, out32, detail8 = net(trick(im, patches, mask_patch))

        if (not use_boundary_2) and (not use_boundary_4) and (not use_boundary_8):
            out, out16, out32 = net(trick(im, patches, mask_patch))
        # 这个就是一个改了一下的交叉熵
        lossp = criteria_p(out, lb)
        loss2 = criteria_16(out16, lb)
        loss3 = criteria_32(out32, lb)
        optim.zero_grad()
        # print(adv_patch_cpu.grad)
        # exit(0)
        boundery_bce_loss = 0.
        boundery_dice_loss = 0.

        if use_boundary_2:
            # if dist.get_rank()==0:
            #     print('use_boundary_2')
            boundery_bce_loss2, boundery_dice_loss2 = boundary_loss_func(detail2, lb)
            boundery_bce_loss += boundery_bce_loss2
            boundery_dice_loss += boundery_dice_loss2

        if use_boundary_4:
            # if dist.get_rank()==0:
            #     print('use_boundary_4')
            boundery_bce_loss4, boundery_dice_loss4 = boundary_loss_func(detail4, lb)
            boundery_bce_loss += boundery_bce_loss4
            boundery_dice_loss += boundery_dice_loss4

        if use_boundary_8:
            # if dist.get_rank()==0:
            #     print('use_boundary_8')
            # 这里计算了一个boundary的loss
            boundery_bce_loss8, boundery_dice_loss8 = boundary_loss_func(detail8, lb)
            boundery_bce_loss += boundery_bce_loss8
            boundery_dice_loss += boundery_dice_loss8

        loss = lossp + loss2 + loss3 + boundery_bce_loss + boundery_dice_loss

        loss.backward()
        optim.step()

        patches[0].data.clamp_(0, 1)
        patches[1].data.clamp_(0, 1)
        patches[2].data.clamp_(0, 1)
        patches[3].data.clamp_(0, 1)

        loss_avg.append(loss.item())

        loss_boundery_bce.append(boundery_bce_loss.item())
        loss_boundery_dice.append(boundery_dice_loss.item())
        # torch.cuda.empty_cache()

        ## print training log message
        if (it + 1) % msg_iter == 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            # lr = optim.lr
            lr = optim.defaults["lr"]
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))

            loss_boundery_bce_avg = sum(loss_boundery_bce) / len(loss_boundery_bce)
            loss_boundery_dice_avg = sum(loss_boundery_dice) / len(loss_boundery_dice)
            msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'boundery_bce_loss: {boundery_bce_loss:.4f}',
                'boundery_dice_loss: {boundery_dice_loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it=it + 1,
                max_it=max_iter,
                lr=lr,
                loss=loss_avg,
                boundery_bce_loss=loss_boundery_bce_avg,
                boundery_dice_loss=loss_boundery_dice_avg,
                time=t_intv,
                eta=eta
            )

            logger.info(msg)
            loss_avg = []
            loss_boundery_bce = []
            loss_boundery_dice = []
            st = ed

            # print(boundary_loss_func.get_params())
        if (it + 1) % save_iter_sep == 0:  # and it != 0:

            ## model
            logger.info('evaluating the model ...')
            logger.info('setup and restore model')

            net.eval()
            # trick.eval()

            # ## evaluator
            eval_times += 1
            logger.info('compute the mIOU')
            with torch.no_grad():
                single_scale1 = MscEvalV0()
                mIOU50 = single_scale1(net, dlval, n_classes, trick, patches, mask_patch)

                single_scale2 = MscEvalV0(scale=0.75)
                mIOU75 = single_scale2(net, dlval, n_classes, trick, patches, mask_patch)

            # save_pth = osp.join(save_pth_path, 'model_iter{}_mIOU50_{}_mIOU75_{}.pth'
            #                     .format(it + 1, str(round(mIOU50, 4)), str(round(mIOU75, 4))))
            # # torch.save(net, save_pth)
            # state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            # # if dist.get_rank()==0:
            # torch.save(state, save_pth)

            # logger.info('training iteration {}, model saved to: {}'.format(it + 1, save_pth))
            logger.info('mIOU50 is: {}, mIOU75 is: {}'.format(mIOU50, mIOU75))

            save_patch_path = "./images/patch/"
            index = ['00', '01', '10', '11']
            for i, p in zip(index, patches):
                patch_name = os.path.join(save_patch_path + i + '/', f"{mIOU75}_{mIOU50}.jpg")
                torchvision.utils.save_image(p, patch_name)
            # summary.add_scalar("mIOU75", mIOU75, eval_times)
            # summary.add_scalar("mIOU50", mIOU50, eval_times)

            # exit(1)
            if mIOU50 > maxmIOU50:
                maxmIOU50 = mIOU50
            save_pth = osp.join(save_pth_path, 'model_maxmIOU50.pth'.format(it + 1))
            # state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            # # if dist.get_rank()==0:
            # torch.save(state, save_pth)
            # torch.save(net, save_pth)
            logger.info('max mIOU model saved to: {}'.format(save_pth))

            if mIOU75 > maxmIOU75:
                maxmIOU75 = mIOU75
            save_pth = osp.join(save_pth_path, 'model_maxmIOU75.pth'.format(it + 1))
            # state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            # if dist.get_rank()==0:
            # torch.save(state, save_pth)
            # torch.save(net, save_pth)
            logger.info('max mIOU model saved to: {}'.format(save_pth))
            logger.info('maxmIOU50 is: {}, maxmIOU75 is: {}.'.format(maxmIOU50, maxmIOU75))

            # net.train()
            net.eval()
    ## dump the final model
    save_pth = osp.join(save_pth_path, 'model_final.pth')
    net.cpu()
    # state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    # if dist.get_rank()==0:
    # torch.save(state, save_pth)
    # torch.save(net, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    print('epoch: ', epoch)


if __name__ == "__main__":
    train()

#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import loss.rmi_utils as rmi_utils
from loss.util import enet_weighing
from utils import semantic_edge
import numpy as np


class SemanticEdgeLoss(nn.Module):
    def __init__(self):
        super(SemanticEdgeLoss, self).__init__()
        self.calculate = nn.BCELoss(reduction='mean')

    def forward(self, pred, mask, class_num=19):
        loss_arr = []
        # mask[mask == 255] = 0
        mask_edge = semantic_edge(mask, class_num)
        pred_edge = semantic_edge(pred.argmax(dim=1), class_num)
        for i in range(class_num):
            loss_arr.append(self.calculate(pred_edge[i].type(torch.FloatTensor).cuda(),
                                           mask_edge[i].type(torch.FloatTensor).cuda()))
        return sum(loss_arr) / len(loss_arr)


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        # 细节，这里用了个负号，因为log后thresh的值是小于0的，所以用负号会变成正数
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        # 先做一个交叉熵
        criteria = nn.CrossEntropyLoss(ignore_index=self.ignore_lb, reduction='none')
        loss1 = criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss1, descending=True)
        # 这里我看不太明白
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class WeightedOhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, num_classes, ignore_lb=255, *args, **kwargs):
        super(WeightedOhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.num_classes = num_classes
        # self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        criteria = nn.CrossEntropyLoss(weight=enet_weighing(labels, self.num_classes).cuda(),
                                       ignore_index=self.ignore_lb, reduction='none')
        loss = criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


_euler_num = 2.718281828  # euler number
_pi = 3.14159265  # pi
_ln_2_pi = 1.837877  # ln(2 * pi)
_CLIP_MIN = 1e-6  # min clip value after softmax or sigmoid operations
_CLIP_MAX = 1.0  # max clip value after softmax or sigmoid operations
_POS_ALPHA = 5e-4  # add this factor to ensure the AA^T is positive definite
_IS_SUM = 1  # sum the loss per channel

__all__ = ['RMILoss']


class RMILoss(nn.Module):
    """
    region mutual information
    I(A, B) = H(A) + H(B) - H(A, B)
    This version need a lot of memory if do not dwonsample.
    """

    def __init__(self,
                 num_classes=21,
                 rmi_radius=3,
                 rmi_pool_way=0,
                 rmi_pool_size=3,
                 rmi_pool_stride=3,
                 loss_weight_lambda=0.5,
                 ignore_lb=0,
                 lambda_way=1):
        super(RMILoss, self).__init__()
        self.num_classes = num_classes
        # radius choices
        assert rmi_radius in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.rmi_radius = rmi_radius
        assert rmi_pool_way in [0, 1, 2, 3]
        self.rmi_pool_way = rmi_pool_way

        # set the pool_size = rmi_pool_stride
        assert rmi_pool_size == rmi_pool_stride
        self.rmi_pool_size = rmi_pool_size
        self.rmi_pool_stride = rmi_pool_stride
        self.weight_lambda = loss_weight_lambda
        self.lambda_way = lambda_way

        # dimension of the distribution
        self.half_d = self.rmi_radius * self.rmi_radius
        self.d = 2 * self.half_d
        self.kernel_padding = self.rmi_pool_size // 2
        # ignore class
        self.ignore_index = ignore_lb

    def forward(self, logits_4D, labels_4D):
        loss = self.forward_sigmoid(logits_4D, labels_4D)
        # loss = self.forward_softmax_sigmoid(logits_4D, labels_4D)
        return loss

    def forward_softmax_sigmoid(self, logits_4D, labels_4D):
        """
        Using both softmax and sigmoid operations.
        Args:
                logits_4D 	:	[N, C, H, W], dtype=float32
                labels_4D 	:	[N, H, W], dtype=long
        """
        # PART I -- get the normal cross entropy loss
        normal_loss = F.cross_entropy(input=logits_4D,
                                      target=labels_4D.long(),
                                      ignore_index=self.ignore_index,
                                      reduction='mean')

        # PART II -- get the lower bound of the region mutual information
        # get the valid label and logits
        # valid label, [N, C, H, W]
        label_mask_3D = labels_4D < self.num_classes
        valid_onehot_labels_4D = F.one_hot(
            labels_4D.long() * label_mask_3D.long(), num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        valid_onehot_labels_4D = valid_onehot_labels_4D * \
                                 label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(
            0, 3, 1, 2).requires_grad_(False)
        # valid probs
        probs_4D = F.sigmoid(logits_4D) * label_mask_3D.unsqueeze(dim=1)
        probs_4D = probs_4D.clamp(min=_CLIP_MIN, max=_CLIP_MAX)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * normal_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else normal_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def forward_sigmoid(self, logits_4D, labels_4D):
        """
        Using the sigmiod operation both.
        Args:
                logits_4D 	:	[N, C, H, W], dtype=float32
                labels_4D 	:	[N, H, W], dtype=long
        """
        # label mask -- [N, H, W, 1]
        # ignore的地方变成0
        label_mask_3D = labels_4D < self.num_classes

        # valid label
        # N,H,W,C
        valid_onehot_labels_4D = F.one_hot(
            labels_4D.long() * label_mask_3D.long(), num_classes=self.num_classes).float()
        label_mask_3D = label_mask_3D.float()
        # nhw
        label_mask_flat = label_mask_3D.view([-1, ])
        # N,H,W,C，去掉ignorelable的影响？
        valid_onehot_labels_4D = valid_onehot_labels_4D * \
                                 label_mask_3D.unsqueeze(dim=3)
        valid_onehot_labels_4D.requires_grad_(False)

        # PART I -- calculate the sigmoid binary cross entropy loss 计算sigmoid的二元交叉熵
        # N*H*W,C
        valid_onehot_label_flat = valid_onehot_labels_4D.view(
            [-1, self.num_classes]).requires_grad_(False)
        # N*H*W,C
        logits_flat = logits_4D.permute(
            0, 2, 3, 1).contiguous().view([-1, self.num_classes])

        # binary loss, multiplied by the not_ignore_mask
        # 求出真正有标签的有多少个像素点
        valid_pixels = torch.sum(label_mask_flat)
        # 求对数的二元交叉熵
        binary_loss = F.binary_cross_entropy_with_logits(logits_flat,
                                                         target=valid_onehot_label_flat,
                                                         weight=label_mask_flat.unsqueeze(
                                                             dim=1),
                                                         reduction='sum')
        bce_loss = torch.div(binary_loss, valid_pixels + 1.0)

        # PART II -- get rmi loss
        # onehot_labels_4D -- [N, C, H, W]
        # 计算MI的下界
        probs_4D = logits_4D.sigmoid() * label_mask_3D.unsqueeze(dim=1) + _CLIP_MIN
        valid_onehot_labels_4D = valid_onehot_labels_4D.permute(
            0, 3, 1, 2).requires_grad_(False)

        # get region mutual information
        rmi_loss = self.rmi_lower_bound(valid_onehot_labels_4D, probs_4D)

        # add together
        final_loss = (self.weight_lambda * bce_loss + rmi_loss * (1 - self.weight_lambda) if self.lambda_way
                      else bce_loss + rmi_loss * self.weight_lambda)

        return final_loss

    def rmi_lower_bound(self, labels_4D, probs_4D):
        """
        calculate the lower bound of the region mutual information.
        Args:
                labels_4D 	:	[N, C, H, W], dtype=float32
                probs_4D 	:	[N, C, H, W], dtype=float32
        """
        assert labels_4D.size() == probs_4D.size()

        p, s = self.rmi_pool_size, self.rmi_pool_stride
        # 下采样
        if self.rmi_pool_stride > 1:
            if self.rmi_pool_way == 0:
                # 用3x3对核进行池化
                labels_4D = F.max_pool2d(
                    labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.max_pool2d(
                    probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 1:
                labels_4D = F.avg_pool2d(
                    labels_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
                probs_4D = F.avg_pool2d(
                    probs_4D, kernel_size=p, stride=s, padding=self.kernel_padding)
            elif self.rmi_pool_way == 2:
                # interpolation
                shape = labels_4D.size()
                new_h, new_w = shape[2] // s, shape[3] // s
                labels_4D = F.interpolate(
                    labels_4D, size=(new_h, new_w), mode='nearest')
                probs_4D = F.interpolate(probs_4D, size=(
                    new_h, new_w), mode='bilinear', align_corners=True)
            else:
                raise NotImplementedError("Pool way of RMI is not defined!")
        # we do not need the gradient of label.
        label_shape = labels_4D.size()
        n, c = label_shape[0], label_shape[1]

        # combine the high dimension points from label and probability map. new shape [N, C, radius * radius, H, W]
        # 获取输入图像和标签在高纬度结合的点
        la_vectors, pr_vectors = rmi_utils.map_get_pairs(
            labels_4D, probs_4D, radius=self.rmi_radius, is_combine=0)

        la_vectors = la_vectors.view(
            [n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor).requires_grad_(False)
        pr_vectors = pr_vectors.view(
            [n, c, self.half_d, -1]).type(torch.cuda.DoubleTensor)

        # small diagonal matrix, shape = [1, 1, radius * radius, radius * radius]
        # 获取对角线矩阵
        diag_matrix = torch.eye(self.half_d).unsqueeze(dim=0).unsqueeze(dim=0)

        # the mean and covariance of these high dimension points
        # Var(X) = E(X^2) - E(X) E(X), N * Var(X) = X^2 - X E(X)
        la_vectors = la_vectors - la_vectors.mean(dim=3, keepdim=True)
        la_cov = torch.matmul(la_vectors, la_vectors.transpose(2, 3))

        pr_vectors = pr_vectors - pr_vectors.mean(dim=3, keepdim=True)
        pr_cov = torch.matmul(pr_vectors, pr_vectors.transpose(2, 3))
        # https://github.com/pytorch/pytorch/issues/7500
        # waiting for batched torch.cholesky_inverse()
        # 计算这个矩阵的逆矩阵
        pr_cov_inv = torch.inverse(
            pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)
        # if the dimension of the point is less than 9, you can use the below function
        # to acceleration computational speed.
        # pr_cov_inv = utils.batch_cholesky_inverse(pr_cov + diag_matrix.type_as(pr_cov) * _POS_ALPHA)

        la_pr_cov = torch.matmul(la_vectors, pr_vectors.transpose(2, 3))
        # the approxiamation of the variance, det(c A) = c^n det(A), A is in n x n shape;
        # then log det(c A) = n log(c) + log det(A).
        # appro_var = appro_var / n_points, we do not divide the appro_var by number of points here,
        # and the purpose is to avoid underflow issue.
        # If A = A^T, A^-1 = (A^-1)^T.
        appro_var = la_cov - \
                    torch.matmul(la_pr_cov.matmul(pr_cov_inv),
                                 la_pr_cov.transpose(-2, -1))
        # appro_var = la_cov - torch.chain_matmul(la_pr_cov, pr_cov_inv, la_pr_cov.transpose(-2, -1))
        # appro_var = torch.div(appro_var, n_points.type_as(appro_var)) + diag_matrix.type_as(appro_var) * 1e-6

        # The lower bound. If A is nonsingular, ln( det(A) ) = Tr( ln(A) ).
        rmi_now = 0.5 * \
                  rmi_utils.log_det_by_cholesky(
                      appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)
        # rmi_now = 0.5 * torch.logdet(appro_var + diag_matrix.type_as(appro_var) * _POS_ALPHA)

        # mean over N samples. sum over classes.
        rmi_per_class = rmi_now.view(
            [-1, self.num_classes]).mean(dim=0).float()
        # is_half = False
        # if is_half:
        #	rmi_per_class = torch.div(rmi_per_class, float(self.half_d / 2.0))
        # else:
        rmi_per_class = torch.div(rmi_per_class, float(self.half_d))

        rmi_loss = torch.sum(
            rmi_per_class) if _IS_SUM else torch.mean(rmi_per_class)
        return rmi_loss


class OhemCrossEntropy2dTensor(nn.Module):
    def __init__(self, ignore_label, reduction='elementwise_mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class CriterionDSN(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255, reduce=True):
        super(CriterionDSN, self).__init__()

        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, preds, target):
        scale_pred = preds[0]
        loss1 = super(CriterionDSN, self).forward(scale_pred, target)
        scale_pred = preds[1]
        loss2 = super(CriterionDSN, self).forward(scale_pred, target)

        return loss1 + loss2 * 0.4


class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the models.
    '''

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=thresh, min_kept=min_kept)
        # self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds, size=(h, w), mode='bilinear', align_corners=True)
        loss1 = self.criterion1(scale_pred, target)

        # scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        # loss2 = self.criterion2(scale_pred, target)

        return loss1


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16 * 20 * 20 // 16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16 * 20 * 20 // 16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()

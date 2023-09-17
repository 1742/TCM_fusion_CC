import sys

import torch
from torch import nn
from torch.functional import F

import numpy as np
from itertools import combinations


class DSNLoss_cls(nn.Module):
    def __init__(self, num: int = 3, weights: [list, tuple, torch.Tensor] = None, criterion: str = 'CE', reduction: str = 'mean'):
        """
        DSN损失，可设置各模态的固定权重

        :param num:
            模态数量
        :param weights:
            各模态权重，大小在0~1，维度为(w1, w2, ...)
        :param criterion:
            损失函数
        :param reduction:
        """
        super(DSNLoss_cls, self).__init__()
        self.num = num

        if weights is None:
            weights = torch.ones(num).float()
        self.weights = torch.Tensor(weights).unsqueeze(0)
        assert self.weights.size(1) == num, \
            'The num of DSN you set was wrong. num {}, but weight length {}'.format(num, len(self.weight))

        if criterion == 'CE':
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.reduction = reduction

    def forward(self, logits: list[torch.Tensor], labels: torch.Tensor):
        assert len(logits) == self.num, \
            'The num of DSN you set was wrong. logits num: {}, expect {}'.format(len(logits), self.num)

        # 根据输入的标签类型转化成序号标签
        if labels.max() == 1 and logits[0].size(1) != 2:
            labels = torch.argmax(labels, dim=1).long().cpu()
        else:
            labels = labels.long().cpu()

        loss = torch.zeros(labels.size(0), self.num)
        for i, logit in enumerate(logits):
            logit = logit.cpu()
            loss[:, i] = self.criterion(logit, labels)

        loss = (self.weights * loss).sum(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class AdaptiveDSNLoss_cls(nn.Module):
    def __init__(self, criterion: str = 'CE', reduction: str = 'mean'):
        """
        自适应DSN，计算时需传入动态权重

        :param criterion:
        :param reduction:
        """
        super(AdaptiveDSNLoss_cls, self).__init__()
        if criterion == 'CE':
            self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.reduction = reduction

    def forward(self, logits: list[torch.Tensor], labels: torch.Tensor, weights: torch.Tensor):
        """
        计算带自适应权重的DSN

        :param logits:
            包含各模态输出logits的列表 -> [logits1, logits2, ...]
        :param labels:
            分类标签
        :param weights:
            自适应动态权重 -> (bs, mode_num)
        :return:
        """
        weights = weights.cpu()

        assert len(logits) == weights.size(1), \
            'The num of DSN you set was wrong. logits num: {}, expect {}'.format(len(logits), weights.size(1))

        # 根据输入的标签类型转化成序号标签
        if labels.max() == 1 and logits[0].size(1) != 2:
            labels = torch.argmax(labels, dim=1).long().cpu()
        else:
            labels = labels.long().cpu()

        loss = torch.zeros(labels.size(0), len(logits))
        for i, logit in enumerate(logits):
            logit = logit.cpu()
            loss[:, i] = self.criterion(logit, labels)

        loss = (weights * loss).sum(1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss





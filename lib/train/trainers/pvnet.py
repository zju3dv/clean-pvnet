"""
trainers.pvnet模块
==================

对PVNet网络进行封装,实现网络末端的损失函数.PVNet中有两类输出,需采用不同的损失函数:

- vote(投票向量):属于线性问题,故采用SmoothL1损失函数
- segment(分割掩码):属于分类问题,故采用CrossEntropy损失函数

"""
# 第三方库
import torch
import torch.nn as nn
# 自建库
from lib.utils import net_utils

class NetworkWrapper(nn.Module):
    """
    NetworkWrapper 网络末端损失函数的实现,

    :param net: 神经网络实例
    :type net: torch.nn.Module实例
    """
    def __init__(self, net):
        """
        __init__ 初始化函数

        :param net: 神经网络实例
        :type net: torch.nn.Module实例
        """
        super(NetworkWrapper, self).__init__()

        self.net = net
        """网络实例"""
        self.vote_crit = torch.nn.functional.smooth_l1_loss  # reduction='mean',beta=1.0
        """投票准则,即计算实际投票值与理想值的损失函数"""
        self.seg_crit = nn.CrossEntropyLoss()
        """分割准则,即计算实际分割值与理想值的损失函数"""

    def forward(self, batch):
        """
        forward 前向函数

        :param batch: 批量数据
        :type batch: tensor
        :return: output(网络输出), loss(总损失=seg_loss+vote_loss), scalar_stats(标量/损失统计), image_stats
        :rtype: dict, float, dict{vote_loss,seg_loss,loss}, dict
        """
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch['meta'].keys():
            loss = torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        # 计算每个投票向量的平均SmoothL1Loss
        weight = batch['mask'][:, None].float()
        vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
        vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss

        # 计算掩码的CrossEtropyLoss
        mask = batch['mask'].long()
        seg_loss = self.seg_crit(output['seg'], mask)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

"""
optimier模块
============

选择并生成相应的神经网络优化器:

- adam(Adaptive Moment Estimation Algorithm,自适应动量估计算法)
- radam(Rectified Adam,修正的Adam)
- sgd(Stochastic Gradient Descent,随机梯度下降)

"""
# 第三方库
import torch
# 自建库
from lib.utils.optimizer.radam import RAdam

optim =  RAdam()

_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}
"""优化器工厂"""


def make_optimizer(cfg, net):
    """
    make_optimizer 选择并生成对应的优化器实例

    :param cfg: 配置管理系统
    :type cfg: CfgNode
    :param net: 神经网络实例
    :type net: torch.nn.Module
    :return: 优化器实例
    :rtype: torch.optim.Optimizer
    """    
    params = []
    """"""
    lr = cfg.train.lr
    """学习率"""
    weight_decay = cfg.train.weight_decay
    """权重衰减率"""

    # 搜集网络中所有需要计算梯度的参数
    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # 选择优化器
    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer

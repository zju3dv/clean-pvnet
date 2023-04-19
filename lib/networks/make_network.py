"""
make_network模块
================
"""
# 标准库
import os
import imp


def make_network(cfg):
    """
    make_network 根据项目的任务类型(pvnet,ct_pvnet,ct),来获取特定的神经网络模型

    :param cfg: 项目配置管理器
    :type cfg: CfgNode
    :return: 指定的神经网络模型实例
    :rtype: torch.nn.Module
    """
    module = '.'.join(['lib.networks', cfg.task])
    path = os.path.join('lib/networks', cfg.task, '__init__.py')
    return imp.load_source(module, path).get_network(cfg)

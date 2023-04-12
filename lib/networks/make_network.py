"""
make_network模块
================
"""
# 标准库
import os
import imp
# 自建库
from .resnet import get_pose_net as get_res
# disable dcn temporarily
#from .pose_dla_dcn import get_pose_net as get_dla_dcn
#from .resnet_dcn import get_pose_net as get_res_dcn
from .linear_model import get_linear_model
from .hourglass import get_large_hourglass_net as get_hg




_network_factory = {
    'res': get_res,
#    'dla': get_dla_dcn,
#    'resdcn': get_res_dcn,
    'linear': get_linear_model,
    'hg': get_hg
}
"""network工厂"""


def get_network(cfg):
    """
    get_network _summary_

    :param cfg: _description_
    :type cfg: _type_
    :return: _description_
    :rtype: _type_
    """
    arch = cfg.network
    heads = cfg.heads
    head_conv = cfg.head_conv
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _network_factory[arch]
    network = get_model(num_layers, heads, head_conv)
    return network


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

"""
make_triner模块
================
"""
# 标准库
import os
import imp
# 自建库
from .trainer import Trainer

def _wrapper_factory(cfg, network):
    module = '.'.join(['lib.train.trainers', cfg.task])
    path = os.path.join('lib/train/trainers', cfg.task+'.py')
    network_wrapper = imp.load_source(module, path).NetworkWrapper(network)
    return network_wrapper


def make_trainer(cfg, network):
    """
    make_trainer _summary_

    :param cfg: _description_
    :type cfg: _type_
    :param network: _description_
    :type network: _type_
    :return: _description_
    :rtype: _type_
    """
    network = _wrapper_factory(cfg, network)
    return Trainer(network)

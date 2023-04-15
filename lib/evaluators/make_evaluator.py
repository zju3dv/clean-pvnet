"""
make_evaluator模块
==================

选择并生成对应的评估器(evaluator)实例
"""
# 标准库
import os
import imp
# 自建库
from lib.datasets.dataset_catalog import DatasetCatalog


def _evaluator_factory(cfg):
    """
    _evaluator_factory 评估器(evaluator)工厂,选择并生成评估器实例

    :param cfg: 配置管理器
    :type cfg: CfgNode
    :return: 评估器实例
    :rtype: Evaluator
    """
    task = cfg.task
    data_source = DatasetCatalog.get(cfg.test.dataset)['id']
    module = '.'.join(['lib.evaluators', data_source, task])
    path = os.path.join('lib/evaluators', data_source, task+'.py')
    evaluator = imp.load_source(module, path).Evaluator(cfg.result_dir)
    return evaluator


def make_evaluator(cfg):
    """
    make_evaluator 调用评估器工厂生成评估器(evaluator)实例

    :param cfg: 配置管理器
    :type cfg: CfgNode
    :return: 评估器实例
    :rtype: Evaluator
    """
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)

"""
collate_batch模块
=================

自制的批数据整理器函数(collator)实例
"""

# 第三方库
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

def ct_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}
    ret.update({'img': default_collate([b['img'] for b in batch])})

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    # detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    wh = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.uint8)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1

    wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
    ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
    ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))

    detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01.float()}
    ret.update(detection)

    return ret


def pvnet_collator(batch):
    """
    pvnet_collator pvnet整理器,(当cfg.task为"pvnet"被调用)

    :param batch: 批数据
    :type batch: list
    :return: 整理后的批数据(不同数据的同种属性被整理到同一list下)
    :rtype: dict
    """    
    if 'pose_test' not in batch[0]['meta']:
        return default_collate(batch)

    inp = np.stack(batch[0]['inp'])
    inp = torch.from_numpy(inp)
    meta = default_collate([b['meta'] for b in batch])
    ret = {'inp': inp, 'meta': meta}

    return ret


_collators = {
    'ct': ct_collator,
    'pvnet': pvnet_collator
}


def make_collator(cfg):
    """
    make_collator 生成批数据整理器(collator)

    :param cfg: 系统配置管理器
    :type cfg: CfgNode类
    :return: 批数据整理器
    :rtype: torch.utils.data.dataloader.default_collate函数或自制的整理器函数
    """
    if cfg.task in _collators:          # 当任务(cfg.task)为"ct"或"pvnet"时，返回自制的整理器函数
        return _collators[cfg.task]
    else:                               # 否则，返回默认的整理器函数
        return default_collate

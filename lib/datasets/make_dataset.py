"""
make_dataset模块
================

将数据集封装为一个可迭代的对象(iterable)，可通过batch_size, shuffle, collate_fn等参数设置每次迭代产生的元素的样式，
也可通过num_works, worker_init_fn是数据集加载器(DataLoader)生成多个子进程并行加载数据。
"""

# 标准库
import os
import time
# 第三方库
import numpy as np
import imp
import torch
import torch.utils.data
# 自建库
from .transforms import make_transforms
from . import samplers
from .dataset_catalog import DatasetCatalog
from .collate_batch import make_collator

#
torch.multiprocessing.set_sharing_strategy('file_system')

# 返回指定的Dataset类引用
def _dataset_factory(data_source, task):
    """
    _dataset_factory 基于数据来源(data_source)和具体的任务(task)，返回相应的Dataset类引用

    :param data_source: 数据来源，即本目录下的custom, linemod, tless, etc.文件夹名称
    :type data_source: str
    :param task: 任务类型，如：pvnet, run, etc.
    :type task: str
    :return: 相应的Dataset类引用
    :rtype: class
    """
    module = '.'.join(['lib.datasets', data_source, task])          # 模块名称
    path = os.path.join('lib/datasets', data_source, task+'.py')    # 模块路劲
    dataset = imp.load_source(module, path).Dataset                 # 基于imp库实现Dataset类的引用
    return dataset

# 返回一个数据集(Dataset)实例
def make_dataset(cfg, dataset_name, transforms, is_train=True):
    """
    make_dataset 返回一个数据集(dataset)实例。基于该实例，可通过下标(index)来对数据集的元素进行访问
    (访问同时也实现了对数据的增强操作)，也可通过len()方法获取数据集的大小。

    :param cfg: 系统配置管理器
    :type cfg: CfgNode类
    :param dataset_name: 数据集名称
    :type dataset_name: str
    :param transforms: 四种数据预处理方法的组合(ToTensor, Normalize, ColorJitter, RandomBlur)
    :type transforms: Compose类
    :param is_train: 是否为训练数据, 默认值为True
    :type is_train: bool
    :return: Datast实例
    :rtype: Dataset子类
    """
    args = DatasetCatalog.get(dataset_name)     # 获取指定数据集的目录信息
    data_source = args['id']
    dataset = _dataset_factory(data_source, cfg.task)
    del args['id']
    if data_source in ['linemod', 'custom']:
        args['transforms'] = transforms
        args['split'] = 'train' if is_train == True else 'test'
    # args['is_train'] = is_train
    dataset = dataset(**args)           # 实例Dataset类
    return dataset

# 返回一个采样器(Sampler)实例
def make_data_sampler(dataset, shuffle):
    """
    make_data_sampler 返回一个采样器(Sampler)实例，两种类型：

    - RandomSampler：随机采样器
    - SequentialSampler：顺序采样器

    :param dataset: Dataset实例
    :type dataset: Dataset子类
    :param shuffle: 是否打乱数据集(是否随机采样)
    :type shuffle: bool
    :return: 采样器实例
    :rtype: Sampler子类
    """
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

# 返回一个批采样器(BatchSampler)实例
def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train):
    """
    make_batch_data_sampler 返回一个批采样器(BatchSampler)实例，两种类型：

    - IterationBasedBatchSampler:基于迭代次数进行批采样(循环进行采样直至到达指定迭代次数)
    - ImageSizeBatchSampler:基于图像尺寸进行批采样(可利用随机生成的图像尺寸进行数据增强/几何变换)

    :param cfg: 系统配置管理器
    :type cfg: CfgNode类
    :param sampler: 采样器实例
    :type sampler: Sampler子类
    :param batch_size: 单次采集的样本数量
    :type batch_size: int
    :param drop_last: 若最后一次采样数量小于batch_size，是否放弃这次采样的数据
    :type drop_last: bool
    :param max_iter: 最大迭代次数
    :type max_iter: int
    :param is_train: 采集的数据是否用于训练
    :type is_train: bool
    :return: 批采样实例
    :rtype: BatchSampler子类
    """

    # 若最大迭代次数(max_iter)非-1，则首先生成一个基于迭代次数的批采样器(IterationBasedBatchSampler)实例
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)
    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, max_iter)

    # 若采样策略是基于图像尺寸('img_size')，则重新生成一个基于图像尺寸的批采样器(ImageSizeBatchSampler)实例
    strategy = cfg.train.batch_sampler if is_train else cfg.test.batch_sampler
    if strategy == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size, drop_last, 256, 480, 640)

    return batch_sampler

# 为每一个DataLoader子进程初始化不同随机数种子
def worker_init_fn(worker_id):
    """
    worker_init_fn 为每一个DataLoader子进程初始化不同的随机数种子

    :param worker_id: 当前子进程的ID(若共有N个子进程，work_id的取值范围为[0,N-1])
    :type worker_id: int
    """
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16)))) # 通过系统时间初始化随机数种子，显然可以保证每个子进程的随机数种子是不同的

# 返回一个数据加载器(DataLoader)实例
def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    """
    make_data_loader 返回一个数据集加载器(DataLoader)

    :param cfg: 系统配置管理器
    :type cfg: CfgNode类
    :param is_train: 是否训练, 默认值为True
    :type is_train: bool
    :param is_distributed: 程序是否是分布式运行, 默认值为False
    :type is_distributed: bool
    :param max_iter: 最大迭代次数, 默认值为-1
    :type max_iter: int
    :return: 数据集加载器实例(DataLoader)
    :rtype: DataLoader类

    .. note:: 当设置DataLoader使用多个子进程加载数据时，由于Linux系统通过fork生成的子进程会继承父进程的所有资源
              (包括numpy随机数生成器的状态)，因此所有子进程中使用的numpy随机操作会输出相同的结果，然而数据增强的过程中
              恰恰需要这种随机化的操作。因此为了避免这个问题，我们需要定义work_init_fn函数(DataLoader的每个子进程在加载数据前
              都会先调用此函数)，在这个函数里面重新初始化numpy种子。

    .. warning:: 上面提到的问题仅在pytorch<1.9的版本中出现，新的pytoch版本中已修复了该Bug。
    """
    if is_train:
        batch_size = cfg.train.batch_size
        shuffle = True
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset

    
    transforms = make_transforms(cfg, is_train)                         # 生成transforms实例
    dataset = make_dataset(cfg, dataset_name, transforms, is_train)     # 生成dataset实例
    sampler = make_data_sampler(dataset, shuffle)                       # 生成sampler实例
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter, is_train) # 生成batch_sampler实例
    num_workers = cfg.train.num_workers                                 # 确定dataloader可调用子进程的数目
    collator = make_collator(cfg)                                       # 确定collator函数
    data_loader = torch.utils.data.DataLoader(                          # 生成dataset实例
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collator,
        worker_init_fn=worker_init_fn
    )

    return data_loader

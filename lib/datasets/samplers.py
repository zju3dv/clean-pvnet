"""
samplers模块
==============

本模块实现了两个BatchSampler类，用于不同的批采样场景：

- ImageSizeBatchSampler:在对数据集进行批采样的过程中，同时随机生成图片的高度和宽度作为batch数据的一部分（需指定随机生成范围和步长）；
- IterationBasedBatchSampler:循环对数据集进行批采样，直至达到指定的采样次数num_iterations。

"""

# 标准库
import math

# 第三方库
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler



class ImageSizeBatchSampler(Sampler):
    """
    ImageSizeBatchSampler 额外随机生成图片大小作为batch数据的一部分

    :param sampler: sampler实例
    :type sampler: Sample子类
    :param batch_size: 单次采样的数据量
    :type batch_size: int
    :param drop_last: 当最后一批数据量小于batch_size时，则将其丢弃(取True时)
    :type drop_last: bool
    :param min_size: 图片高宽的最小值, 默认值为600
    :type min_size: int
    :param max_height: 图片高度的最大值, 默认值为800
    :type max_height: int
    :param max_width: 图片宽度的最大值, 默认值为800
    :type max_width: int
    :param size_int: 随机生成步长, 默认值为8
    :type size_int: int
    """
    def __init__(self, sampler, batch_size, drop_last, min_size=600, max_height=800, max_width=800, size_int=8):
        """
        __init__ 初始化函数

        :param sampler: Sampler实例
        :type sampler: Sample子类
        :param batch_size: 单次采样的数据量
        :type batch_size: int
        :param drop_last: 当最后一批数据量小于batch_size时，则将其丢弃(取True时)
        :type drop_last: bool
        :param min_size: 图片高宽的最小值, 默认值为600
        :type min_size: int
        :param max_height: 图片高度的最大值, 默认值为800
        :type max_height: int
        :param max_width: 图片宽度的最大值, 默认值为800
        :type max_width: int
        :param size_int: 随机生成步长, 默认值为8
        :type size_int: int
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.hmin = min_size
        self.hmax = max_height
        self.wmin = min_size
        self.wmax = max_width
        self.size_int = size_int
        self.hint = (self.hmax-self.hmin)//self.size_int+1
        self.wint = (self.wmax-self.wmin)//self.size_int+1

    def generate_height_width(self):
        """
        generate_height_width 在指定范围内：(wmin, wmax, size_int)/(hmin, hmax, size_int)，随机生成高度和宽度

        :return: 返回随机生成的高度和宽度
        :rtype: tuple(int, int)
        """
        hi, wi = np.random.randint(0, self.hint), np.random.randint(0, self.wint)
        h, w = self.hmin + hi * self.size_int, self.wmin + wi * self.size_int
        return h, w

    def __iter__(self):
        """
        __iter__ 将该类转化为一个可迭代对象

        :yield: 可迭代对象
        :rtype: iterator
        """
        batch = []
        h, w = self.generate_height_width()
        for idx in self.sampler:
            batch.append((idx, h, w))
            if len(batch) == self.batch_size:
                h, w = self.generate_height_width()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        """
        __len__ 使len()适用于该类

        :return: 类的长度
        :rtype: int
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class IterationBasedBatchSampler(BatchSampler):
    """
    IterationBasedBatchSampler Wraps a BatchSampler, resampling from it until a specified number of iterations have been sampled
    对BatchSampler再封装，使该类可以循环迭代直至达到指定次数(num_iterations)

    :param batch_sampler: BatchSampler实例
    :type batch_sampler: BatchSampler类
    :param num_iterations: 最大迭代次数
    :type num_iterations: int
    :param start_iter: 起始迭代次数, 默认值为0
    :type start_iter: int
    """
    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        """
        __init__ 初始化函数

        :param batch_sampler: BatchSampler实例
        :type batch_sampler: BatchSampler类
        :param num_iterations: 最大迭代次数
        :type num_iterations: int
        :param start_iter: 起始迭代次数, 默认值为0
        :type start_iter: int
        """
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        """
        __iter__ 将该类转化为一个可迭代对象

        :yield: 可迭代对象
        :rtype: iterator
        """
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        """
        __len__ 使len()适用于该类

        :return: 类的长度
        :rtype: int
        """
        return self.num_iterations

"""
transforms模块
================

本模块主要实现数据增强和预处理的功能：

- ToTensor:归一化（数据增强）;
- Normalize:标准化（数据增强）;
- ColorJitter:随机颜色空间变换（预处理）;
- RandomBlur:随机滤波（预处理）。

四个函数基于Compose类封装在一起，对输入图像的执行顺序为：随机滤波(RandomBlur, 仅训练时使用)、随机颜色空间变换(ColorJitter, 仅训练时使用)、归一化(ToTensor)、标准化(Normalize,均值标准差已预先设定)处理。
"""

# 标准库
import random

# 第三方库
import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
from PIL import Image


class Compose(object):
    """
    Compose ToTensor, Normalize, ColorJitter, RandomBlur的封装，并依次对图片进行随机滤波、随机颜色空间变换、归一化、标准化处理。

    :param transforms: ToTensor, Normalize, ColorJitter, RandomBlur的实例
    :type transforms: list
    """
    def __init__(self, transforms):
        """
        __init__ 初始化函数

        :param transforms: ToTensor, Normalize, ColorJitter, RandomBlur的实例
        :type transforms: list
        """
        self.transforms = transforms

    def __call__(self, img, kpts=None, mask=None):
        """
        __call__ __call__ 调用函数(使类可以像函数一样被调用)

        :param image: 输入图像
        :type image: PIL.Image类
        :param kpts: 2D关键点
        :type kpts: narray类
        :param mask: 图像掩码
        :type mask: narray类
        :return: img, kpts, mask
        :rtype: tuple
        """
        # 依次对输入图片进行随机滤波、随机颜色空间变换、归一化、标准化处理。
        for t in self.transforms:
            img, kpts, mask = t(img, kpts, mask)
        return img, kpts, mask

    def __repr__(self):
        """
        __repr__ represtentation函数

        :return: 类的描述字符串
        :rtype: str
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    """
    ToTensor 将图像数据归一化
    """
    def __call__(self, img, kpts, mask):
        """
        __call__ 调用函数(使类可以像函数一样被调用)

        :param image: 输入图像
        :type image: PIL.Image类
        :param kpts: 2D关键点
        :type kpts: narray类
        :param mask: 图像掩码
        :type mask: narray类
        :return: img, kpts, mask
        :rtype: tuple
        """
        return np.asarray(img).astype(np.float32) / 255., kpts, mask


class Normalize(object):
    """
    Normalize 将图像标准化处理,并将图像的通道放在第一个维度,(H,W,C)->(C,H,W)。

        :param mean: 图像均值
        :type mean: _type_
        :param std: 图像标准差
        :type std: _type_
        :param to_bgr: 是否将图像转化为BGR格式, 默认值为True
        :type to_bgr: bool
    """
    def __init__(self, mean, std, to_bgr=True):
        """
        __init__ 初始化函数

        :param mean: 图像均值
        :type mean: _type_
        :param std: 图像标准差
        :type std: _type_
        :param to_bgr: 是否将图像转化为BGR格式, 默认值为True
        :type to_bgr: bool
        """
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, kpts, mask):
        """
        __call__ 调用函数(使类可以像函数一样被调用)

        :param image: 输入图像
        :type image: PIL.Image类
        :param kpts: 2D关键点
        :type kpts: narray类
        :param mask: 图像掩码
        :type mask: narray类
        :return: img, kpts, mask
        :rtype: tuple
        """
        # 标准化处理
        img -= self.mean
        img /= self.std
        if self.to_bgr:     # (H,W,C)->(C,H,W)
            img = img.transpose(2, 0, 1).astype(np.float32)
            # img = img.transpose(2, 1, 0).astype(np.float32)
        return img, kpts, mask


class ColorJitter(object):
    """
    ColorJitter 对输入图像进行颜色空间的变换

    :param brightness: 亮度, 默认值为None
    :type brightness: float or tuple of float (min, max)
    :param contrast: 对比度, 默认值为None
    :type contrast: float or tuple of float (min, max)
    :param saturation: 饱和度, 默认值为None
    :type saturation: float or tuple of float (min, max)
    :param hue: 色调, 默认值为None
    :type hue: float or tuple of float (min, max)
    """
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        """
        __init__ 初始化函数

        :param brightness: 亮度, 默认值为None
        :type brightness: float or tuple of float (min, max)
        :param contrast: 对比度, 默认值为None
        :type contrast: float or tuple of float (min, max)
        :param saturation: 饱和度, 默认值为None
        :type saturation: float or tuple of float (min, max)
        :param hue: 色调, 默认值为None
        :type hue: float or tuple of float (min, max)
        """
        # 返回一个ColorJitter实例
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image, kpts, mask):
        """
        __call__ 调用函数(使类可以像函数一样被调用)

        :param image: 输入图像
        :type image: PIL.Image类
        :param kpts: 2D关键点
        :type kpts: narray类
        :param mask: 图像掩码
        :type mask: narray类
        :return: img, kpts, mask
        :rtype: tuple
        """
        # 先将image调整为内存连续储存，再将其转换为PIL格式，最后进行颜色空间的变换
        image = np.asarray(self.color_jitter(Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, kpts, mask


class RandomBlur(object):
    """
    RandomBlur 随机对图片进行滤波处理

    :param prob: 图片被处理的概率，默认值为0.5
    :type prob: float
    """
    def __init__(self, prob=0.5):
        """
        __init__ 初始化函数

        :param prob: 图片被处理概率，默认值为0.5
        :type prob: float
        """
        self.prob = prob

    def __call__(self, image, kpts, mask):
        """
        __call__ 调用函数(使类可以像函数一样被调用)

        :param image: 输入图像
        :type image: PIL.Image类
        :param kpts: 2D关键点
        :type kpts: narray类
        :param mask: 图像掩码
        :type mask: narray类
        :return: img, kpts, mask
        :rtype: tuple
        """
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])              # 随机生成σ:3, 5, 7, 9
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)  # 对图片进行高斯滤波，去除噪声
        return image, kpts, mask


def make_transforms(cfg, is_train):
    """
    make_transforms _summary_

    :param cfg: 系统配置管理器
    :type cfg: CfgNode类
    :param is_train: 是否训练
    :type is_train: bool
    :return: 四个用于图像变换的函数组合
    :rtype: Compose类实例
    """
    if is_train is True:
        transform = Compose(
            [
                RandomBlur(0.5),
                ColorJitter(0.1, 0.1, 0.05, 0.05),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform

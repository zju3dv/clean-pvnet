"""
pvnet模块
=========

主要用pvnet类对linemode数据集进行封装，并实现了以下功能：

- __iter__方法：可通过索引(index)来linemode数据集中的对应数据进行访问
- __len__方法：可通过调用len()方法来获取数据集的数据总量
- 数据增强：对数据进行颜色空间、几何空间、像素空间的随机变换
"""

# 标准库
import os
import random

# 第三方库
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# 自建库
from lib.config import cfg
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1

class Dataset(data.Dataset):
    """
    Dataset 封装Linemod数据集，实现数据增强、__iter__、__len__等功能或方法

    :param ann_file: linemod属性文件路径
    :type ann_file: str
    :param data_root: linemod图像路径
    :type data_root: str
    :param split: 这里是是否训练的标志
    :type split: str
    :param transforms: transform方法, 默认值为None
    :type transforms: transforms.Compose类
    """
    def __init__(self, ann_file, data_root, split, transforms=None):
        """
        __init__ 初始化函数

        :param ann_file: linemod属性文件路径
        :type ann_file: str
        :param data_root: linemod图像路径
        :type data_root: str
        :param split: 这里是是否训练的标志
        :type split: str
        :param transforms: transform方法, 默认值为None
        :type transforms: transforms.Compose类
        """
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))
        self._transforms = transforms
        self.cfg = cfg

    def read_data(self, img_id):
        """
        read_data 根据图像id读取对应的图像，2D关键点和掩码

        :param img_id: 图像id
        :type img_id: int
        :return: 对应的图像，2D关键点和掩码
        :rtype: tuple(PIL.Image, narray(9*2), narray)
        """        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        inp = Image.open(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        cls_idx = linemod_config.linemod_cls_names.index(anno['cls']) + 1
        mask = pvnet_data_utils.read_linemod_mask(anno['mask_path'], anno['type'], cls_idx)

        return inp, kpt_2d, mask

    def __getitem__(self, index_tuple):
        """
        __getitem__ _summary_

        :param index_tuple: 索引值
        :type index_tuple: tuple(index, heigit, weight)
        :return: 图像img，掩码mask，像素指向特征点的单位向量vertex，图像id，空字典meta
        :rtype: dict
        """                
        index, height, width = index_tuple
        img_id = self.img_ids[index]

        img, kpt_2d, mask = self.read_data(img_id)
        if self.split == 'train':
            # 若正在取训练数据，则需要对数据进行增强
            inp, kpt_2d, mask = self.augment(img, mask, kpt_2d, height, width)
        else:
            inp = img

        if self._transforms is not None:
            # 若存在图像变换(transform)实例，则利用该实例对图像进行变换
            inp, kpt_2d, mask = self._transforms(inp, kpt_2d, mask)
        
        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1) # 计算各个像素指向特征点的单位向量
        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex, 'img_id': img_id, 'meta': {}}
        # visualize_utils.visualize_linemod_ann(torch.tensor(inp), kpt_2d, mask, True)

        return ret

    def __len__(self):
        """
        __len__ 使len()方法对该类可用

        :return: 数据集的数据总量
        :rtype: int
        """        
        return len(self.img_ids)

    def augment(self, img, mask, kpt_2d, height, width):
        """
        augment 对输入图像进行几何空间的变换(旋转)，以及增加截断效果(剪裁)，最终实现数据增强的目的

        :param img: 输入图像
        :type img: PIL.
        :param mask: 掩码
        :type mask: narray
        :param kpt_2d: 2D关键点
        :type kpt_2d: narray
        :param height: 输出图像的高度
        :type height: int
        :param width: 输出图像的宽度
        :type width: int
        :return: 图像，掩码，齐次坐标
        :rtype: tuple(img, mask, hcoords)
        """        
        # add one column to kpt_2d for convenience to calculate
        hcoords = np.concatenate((kpt_2d, np.ones((9, 1))), axis=-1) # size(9*2) -> size(9*3)，转换为齐次坐标(homogeneous coordinate)
        img = np.asarray(img).astype(np.uint8)
        foreground = np.sum(mask)
        # randomly mask out to add occlusion
        if foreground > 0:      # 图像中存在目标实例
            img, mask, hcoords = rotate_instance(img, mask, hcoords, self.cfg.train.rotate_min, self.cfg.train.rotate_max)
            img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width,
                                                         self.cfg.train.overlap_ratio,
                                                         self.cfg.train.resize_ratio_min,
                                                         self.cfg.train.resize_ratio_max)
        else:                   # 图像中不存在目标实例
            img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)
        kpt_2d = hcoords[:, :2]

        return img, kpt_2d, mask

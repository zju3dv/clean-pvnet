"""
evaluators.custom.pvnet模块
===========================

本模块主要实现了一个评估器类(Evaluator),该类集成了用于计算各项评估指标的功能函数:

- ``2D-proj`` 2D投影误差指标 衡量位姿估计: 将目标模型投影在预测图像平面和实际图像平面上,计算两图像上模型对应像素点的平均距离(误差).
  通常情况下当误差小于5个像素时,则认为位姿预测正确.
- ``ADD(-S)`` 平均模型点指标 衡量位姿估计: 计算预测模型3D点和实际模型3D点间的平均距离(误差).通常情况下,当误差小于模型直径的10%时,
  则认为位姿预测正确.
- ``5cm5°`` 5厘米5度指标 衡量位姿估计: 计算预测位姿实际位姿的平移误差和旋转误差.当平移误差小于5cm,旋转误差小于5°时,则认为位姿预测正确.
- ``IoU`` 区域交集指标 衡量实例分割: 计算预测实例分割区域和实际的实例分割区域的交集和并集,用交集除以并集的结果来作为实例分割的衡量指标.

"""
# 标准库
import os
# 第三方库
import torch
import numpy as np
import pycocotools.coco as coco
from PIL import Image
from scipy import spatial
# 自建库
from lib.config import cfg
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.utils.img_utils import read_depth
from lib.utils.linemod import linemod_config
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
if cfg.test.icp:
    from lib.utils import icp_utils

class Evaluator:
    """
    Evaluator 评估神经网络对位姿估计和实例分割的预测结果.对于位姿估计,评估了2D-proj(2D投影指标),ADD(-S)(平均最近点指标),
    5cm5°三个方面的衡量指标.对于实例分割,采用IoU衡量指标.

    :param result_dir: 保存评估结果的文件路径
    :type result_dir: str   
    """    
    def __init__(self, result_dir):
        """
        __init__ 初始化函数

        :param result_dir: 保存评估结果的文件路径
        :type result_dir: str
        """
        self.result_dir = result_dir
        """保存评估结果文件路径"""
        args = DatasetCatalog.get(cfg.test.dataset)
        """测试集的路径信息"""
        self.ann_file = args['ann_file']
        """测试集的标注文件路径"""
        self.coco = coco.COCO(self.ann_file)
        """测试集的标注文件实例化对象"""

        data_root = args['data_root']
        """测试集的文件夹根目录"""
        model_path = data_root + '/model.ply'
        """待测目标的三维模型路径"""
        self.model = pvnet_data_utils.get_ply_model(model_path)
        """目标模型的空间点坐标"""
        self.diameter = np.loadtxt('data/custom/diameter.txt').item()
        """目标模型的直径"""

        self.proj2d = []
        """位姿估计评估结果(2D投影指标)"""
        self.add = []
        """位姿估计评估结果(平均点距离指标)"""
        self.icp_add = []
        """位姿估计评估结果(平均最近点距离指标)"""
        self.cmd5 = []
        """位姿估计评估结果(5cm 5°指标)"""
        self.mask_ap = []
        """实例分割评估结果(IoU指标)"""
        self.icp_render = icp_utils.SynRenderer(cfg.cls_type) if cfg.test.icp else None

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        """
        projection_2d ``2D投影指标`` :对比预测模型和实际的像素点坐标,判断像素点的平均距离是否超过给定阈值(threshold),
        并记录本次评估的结果至proj2d列表容器内.

        :param pose_pred: 预测的目标位姿
        :type pose_pred: narray(3*4)
        :param pose_targets: 实际的目标位姿
        :type pose_targets: narray(3*4)
        :param K: 相机内参
        :type K: narray(3*3)
        :param threshold: 预测误差阈值(单位为像素), 默认值为5
        :type threshold: int
        """
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)  # 基于预测的位姿,计算目标在图片上的像素坐标
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)  # 基于真实的位姿,计算目标在图片上的像素坐标
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))  # 计算每个预测的像素坐标和实际的像素坐标的欧式距离

        self.proj2d.append(proj_mean_diff < threshold)  # 若像素点的平均欧氏距离小于设定的阈值(threshold),则判断其预测正确True,否则预测错误False

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        """
        add_metric ``平均模型点距离指标`` :对比预测模型和实际的空间点坐标,判断空间点的平均距离是否在误差容许范围内
        (diameter*percentage),并记录本次评估的结果至相应的容器内(add 或 icp_add).

        :param pose_pred: 预测的目标位姿
        :type pose_pred: narray(3*4)
        :param pose_targets: 实际的目标位姿
        :type pose_targets: narray(3*4)
        :param icp: 是否将判断结果保存至icp_add内, 默认值为False
        :type icp: bool
        :param syn: 是否基于最近点求距离, 默认值为False
        :type syn: bool
        :param percentage: 容许误差的比例, 默认值为0.1
        :type percentage: float

        .. note:: 本函数实际上实现了两种位姿估计评估指标:

                  - ``ADD-S`` :平均最近点距离指标,适用于对称物体,启用标志:syn==Flase
                  - ``ADD`` :平均最近点指标,适用于非对称物体,启用标志:syn==True
        """
        diameter = self.diameter * percentage  # 预测模型空间点坐标误差容许范围 = 模型直径(diameter) * 比例(percentage)
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)  # 基于模型空间点坐标生成K维树,便于快速查询最近点
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)  # 返回预测模型三维坐标与实际模型三维坐标的最近距离
            mean_dist = np.mean(mean_dist)  # 计算模型所有三维坐标的均值
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))  # 最近距离取对应空间点的欧氏距离

        # 将预测结果保存至icp_add或add列表内(若预测空间点距实际空间点的平均距离在误差容许范围内,则判定预测正确)
        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        """
        cm_degree_5_metric ``5cm 5°指标`` :计算预测位姿和实际位姿的平移误差和旋转误差,若平移误差小于5cm且旋转误差小于5°时,
        判断预测结果正确,并将结果保存至是cmd5列表容器内.

        :param pose_pred: 预测位姿
        :type pose_pred: narray(3*4)
        :param pose_targets: 实际位姿
        :type pose_targets: narray(3*4)
        """
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100  # 计算预测位姿和真实位姿间的平移误差
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))  # 计算两位姿间的旋转误差
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)  # 判断预测结果是否正确,并保存结果

    def mask_iou(self, output, batch):
        """
        mask_iou ``IoU指标`` :计算掩码的IoU(Intersectin over Union)指标 ``(mask_pred ∩ mask_gt)/(mask_pred ∪ mask_gt)`` (mask_pred
        为预测掩码,mask_gt为目标掩码),并将评估结果(iou > 0.7)保存在mask_ap容器列表内

        :param output: 神经网络输出
        :type output: dict
        :param batch: 输入网络的批数据
        :type batch: dict
        """
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()  # 生成预测的掩码mask_pred
        mask_gt = batch['mask'][0].detach().cpu().numpy()  # 导入目标掩码mask_gt
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()  # 计算IoU指标
        self.mask_ap.append(iou > 0.7)  # 若IoU大于0.7,则判断预测的掩码正确

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000
        R_refined, t_refined = icp_utils.icp_refinement(depth, self.icp_render, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), (depth.shape[1], depth.shape[0]), depth_only=True,            max_mean_dist_factor=5.0)
        R_refined, _ = icp_utils.icp_refinement(depth, self.icp_render, R_refined, t_refined, K.copy(), (depth.shape[1], depth.shape[0]), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))
        return pose_pred

    def evaluate(self, output, batch):
        """
        evaluate 评估网络输出的结果.对于位姿估计,采用2D-proj,ADD(-S),5cm5°三项衡量指标.对于实例分割,采用IoU衡量指标.
        并将这些指标保存在实例的属性容器内.

        :param output: 神经网络输出
        :type output: dict
        :param batch: 输入网络的批数据
        :type batch: dict
        """
        # 导入相关参数
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        # 计算2D-proj,ADD(-S),5cm5°,IoU等衡量指标
        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        if self.icp_render is not None:
            pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
            self.add_metric(pose_pred_icp, pose_gt, icp=True)
        self.projection_2d(pose_pred, pose_gt, K)  # 计算2D投影误差衡量指标
        if cfg.cls_type in ['eggbox', 'glue']:  # 对于对称或局部对称的物体采用ADD-S评价指标
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:                                   # 对于非对称物体采用ADD评价指标
            self.add_metric(pose_pred, pose_gt)
        self.cm_degree_5_metric(pose_pred, pose_gt)  # 计算5cm5°衡量指标
        self.mask_iou(output, batch)  # 计算目标检测的IoU指标

    def summarize(self):
        """
        summarize 格式化输出并返回各指标的评估均值

        :return: 各指标的评估均值
        :rtype: dict
        """
        # 计算各指标的均值
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)

        # 格式化输出评估结果
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        if self.icp_render is not None:
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
        
        # 清空保存评估指标的容器
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_add = []

        # 返回评估结果
        return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap}

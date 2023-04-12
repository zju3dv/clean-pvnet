"""
resnet18模块
============

本模块主要基于 ``class Resnet18`` 对PVNet网络模型进行组织和管理.
"""
# 第三方库
import torch
from torch import nn
from torch.nn import functional as F
# 自建库
from .resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, estimate_voting_distribution_with_mean
from lib.config import cfg


class Resnet18(nn.Module):
    """
    Resnet18 构建PVNet网络模型

    :param ver_dim: vertex维度
    :type ver_dim: int
    :param seg_dim: segmentation维度
    :type seg_dim: int
    :param fcdim: 全连接神经层的输出维度, 默认值为256
    :type fcdim: int
    :param s8dim: 8步长神经层的输出维度, 默认值为128
    :type s8dim: int
    :param s4dim: 4步长神经层的输出维度, 默认值为64
    :type s4dim: int
    :param s2dim: 2步长神经层的输出维度, 默认值为32
    :type s2dim: int
    :param raw_dim: raw神经层的输出维度, 默认值为32
    :type raw_dim: int
    """
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        """
        __init__ 初始化函数

        :param ver_dim: vertex维度
        :type ver_dim: int
        :param seg_dim: segmentation维度
        :type seg_dim: int
        :param fcdim: 全连接神经层的输出维度, 默认值为256
        :type fcdim: int
        :param s8dim: 8步长神经层的输出维度, 默认值为128
        :type s8dim: int
        :param s4dim: 4步长神经层的输出维度, 默认值为64
        :type s4dim: int
        :param s2dim: 2步长神经层的输出维度, 默认值为32
        :type s2dim: int
        :param raw_dim: raw神经层的输出维度, 默认值为32
        :type raw_dim: int
        """
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim=ver_dim
        """vertex维度=特征点数量*2"""
        self.seg_dim=seg_dim
        """segmentation维度=2(实例分割的目标个数为1)"""

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, kernel_size=3, stride= 1, padding=1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s
        """原始resnet18模块"""

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        """8步长卷积层"""
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        """二次线性插值上采样层:8步长->4步长"""

        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        """4步长卷积层"""
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)
        """二次线性插值上采样:4步长->2步长"""

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        """2步长卷积层"""
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)
        """二次线性插值上采样:2步长->1步长"""

        self.convraw = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True),
            nn.Conv2d(raw_dim, seg_dim+ver_dim, 1, 1)
        )
        """raw卷积层/原始(raw)卷积层"""

    def _normal_initialization(self, layer):
        """
        _normal_initialization 初始化神经层layer的权重(正态分布)和偏置(初始化为0)

        :param layer: 需要初始化参数的神经层
        :type layer: torch.nn.Module子类
        """
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def decode_keypoint(self, output):
        """
        decode_keypoint 根据PVNet模型的输出计算掩码及特征点坐标

        :param output: 将mask和kpt_2d更新到 ``output`` 字典内
        :type output: dict
        """
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape  # 提取批数量,图片高度,图片宽度,特征向量
        vertex = vertex.view(b, h, w, vn_2//2, 2)  # 分离每个像素点上的9个特征向量
        mask = torch.argmax(output['seg'], 1)  # 提取掩码
        if cfg.test.un_pnp:  # 若为非确定型PnP,只需要计算特征点及其空间概率分布
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
            output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
        else:                # 若为普通型PnP,仅需基于RANSAC算法计算特征点的具体坐标
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
            output.update({'mask': mask, 'kpt_2d': kpt_2d})

    def forward(self, x, feature_alignment=False):    
        """
        forward 基于resnet的PVNet网络组织形式

        :return: 分割预测(掩码)和顶点预测(从各个像素指向关键点的向量)
        :rtype: dict

        .. note:: ``feature_alignment`` 参数在本函数中并未使用
        """
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        fm=self.conv8s(torch.cat([xfc,x8s],1))
        fm=self.up8sto4s(fm)
        if fm.shape[2]==136:
            fm = nn.functional.interpolate(fm, (135,180), mode='bilinear', align_corners=False)

        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)

        x=self.convraw(torch.cat([fm,x],1))
        seg_pred=x[:,:self.seg_dim,:,:]
        ver_pred=x[:,self.seg_dim:,:,:]

        ret = {'seg': seg_pred, 'vertex': ver_pred}

        if not self.training:  # 若当前不是正在训练,则调用decode_keypoint函数,基于RANSAC求出关键点的坐标
            with torch.no_grad():
                self.decode_keypoint(ret)

        return ret


def get_res_pvnet(ver_dim, seg_dim):
    """
    get_res_pvnet 获取基于resnet18的PVNet网络

    :param ver_dim: vertex维度(用于获取特征点)
    :type ver_dim: int
    :param seg_dim: segmentation维度(用于目标检测)
    :type seg_dim: int
    :return: 神经网络模型
    :rtype: torch.nn.Module实例
    """
    model = Resnet18(ver_dim, seg_dim)
    return model


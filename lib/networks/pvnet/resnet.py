"""
resnet模块
==========

本模块主要用于生成经典的ResNet网络.ResNet网络有多种可选的层数(resnet18, resnet34, resnet50, etc.),
但它们具有相同的结构(由ResNet类管理):

1. 初始卷积层:由卷积层,batchnorm层,ReLU激活函数和最大池化层构成.在不同种resnet中保持不变
2. 残差层一:由数个同种残差块(BasicBlock或Bottleneck)串联而成.在不同种resnet中具有不同的结构
3. 残差层二:同上
4. 残差层三:同上
5. 残差层四:同上
6. 全连接层:由平均池化层和全连接层构成.在不同种resnet中保持不变

.. note:: 本模块基于torchvision.models.resnet模块编写.因此本模块的resnet实现方法和官方模块的有所不同,如:官方resnet的最后一层为全连接层,
          但本模块可通过 ``fully_conv`` 设定最后一层为卷积层,从而变resnet为全卷积神经网络.

.. note:: ``Bottleneck`` 残差块与 ``BasicBlock`` 残差块的异同:

          1. BasicBlock中expansion=1, BottleNeck中expansion=4。
          2. BasicBlock模块包含2个卷积层(3x3,3x3),而BottleNeck模块包含3个卷积层(1x1, 3x3, 1x1)。
          3. 在两个模块中都有downsample,作用是改变identity的通道数,使之与out的通道数相匹配。

          Bottleneck残差块主要是为了提高模型的训练速度而设计的.相较于BasicBlock残差块,它对并没有提高模型的性能,
          甚至有所降低.因此Bottleneck主要在较深的resnet中使用,如:resnet50,resnet101, etc.
"""
# 标准库
import math
# 第三方库
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

"""供外部调用的接口"""
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

"""预训练resnet网络参数的下载地址"""
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """
    conv3x3 生成一个3*3大小的卷积核(输入神经元等宽填充)

    :param in_planes: 输入通道数(特征数量)
    :type in_planes: int
    :param out_planes: 输出通道数(特征数量)
    :type out_planes: int
    :param stride: 步长, 默认值为1
    :type stride: int
    :param dilation: 膨胀率, 默认值为1
    :type dilation: int
    :return: 3*3卷积核
    :rtype: torch.nn.Cov2d实例
    """
    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size  # 计算膨胀卷积核的大小

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2  # 保证等宽卷积时,输入神经层两端的填充大小

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)  # 转化为元组类型

    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,  # 生成卷积核实例
                     padding=full_padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    """
    BasicBlock 基本残差模块:由两个卷积层和一条直连边构成.每个卷积层后均有一个batchnorm层.激活函数为ReLU.

    :param inplanes: 输入特征数量
    :type inplanes: int
    :param planes: 期望输出特征数量
    :type planes: int
    :param stride: 步长(仅在本模块初始层使用), 默认值为1
    :type stride: int
    :param downsample: 下采样池化层, 默认值为None
    :type downsample: torch.nn.Sequential
    :param dilation: 膨胀率(在本模块的各个神经层均有使用), 默认值为1
    :type dilation: int
    """
    """输出维度的扩大倍数,本模块的输出维度=expansion*planes"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        """
        __init__ 初始化函数

        :param inplanes: 输入特征数量
        :type inplanes: int
        :param planes: 期望输出特征数量
        :type planes: int
        :param stride: 步长(仅在本模块初始层使用), 默认值为1
        :type stride: int
        :param downsample: 下采样池化层, 默认值为None
        :type downsample: torch.nn.Sequential
        :param dilation: 膨胀率(在本模块的各个神经层均有使用), 默认值为1
        :type dilation: int
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        forward 模块中神经层的组织形式
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck 瓶颈残差块:由三个卷积层构成.每个卷积层后均有一个batchnorm层.激活函数为ReLU.

    :param inplanes: 输入特征数量
    :type inplanes: int
    :param planes: 期望输出特征数量
    :type planes: int
    :param stride: 步长(仅在本模块初始层使用), 默认值为1
    :type stride: int
    :param downsample: 下采样池化层, 默认值为None
    :type downsample: torch.nn.Sequential
    :param dilation: 膨胀率(在本模块的各个神经层均有使用), 默认值为1
    :type dilation: int
    """
    """本模块的输出维度=expansion*4"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        """
        __init__ 初始化函数

        :param inplanes: 输入特征数量
        :type inplanes: int
        :param planes: 期望输出特征数量
        :type planes: int
        :param stride: 步长(仅在本模块初始层使用), 默认值为1
        :type stride: int
        :param downsample: 下采样池化层, 默认值为None
        :type downsample: torch.nn.Sequential
        :param dilation: 膨胀率(在本模块的各个神经层均有使用), 默认值为1
        :type dilation: int
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)

        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        forward 模块中神经层的组织形式
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 组织并管理基本的resnet网络结构.

    :param block: 残差块的类型:基础残差块(BasicBlock)和瓶颈残差块(BottleneckBlock)
    :type block: BasicBlock类或Bottleneck类
    :param layers: resnet网络中四个残差层分别包含残差块的个数
    :type layers: list(int, int, int, int)
    :param num_classes: resnet输出层神经元的数量, 默认值为1000
    :type num_classes: int
    :param fully_conv: 是否采用全卷积神经网络, 默认值为False
    :type fully_conv: bool
    :param remove_avg_pool_layer: 是否移除平均池化层, 默认值为False
    :type remove_avg_pool_layer: bool
    :param output_stride: resnet的最大步长, 默认值为32
    :type output_stride: int

    .. note:: - ``remove_avg_pool_layer`` 参数仅当 ``full_conv = True`` 时生效.
              - ``stride`` 参数表示整个resnet网络的各个神经层的最大累积步长.resnet网络固定最小步长为4,固定最大步长为32.所以
                ``4<=strides<=32''
    """
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 fully_conv=False,
                 remove_avg_pool_layer=False,
                 output_stride=32):
        """
        __init__ 初始化函数

        :param block: 残差块的类型:基础残差块(BasicBlock)和瓶颈残差块(BottleneckBlock)
        :type block: BasicBlock类或Bottleneck类
        :param layers: resnet网络中四个残差层分别包含残差块的个数
        :type layers: list(int, int, int, int)
        :param num_classes: resnet输出层神经元的数量, 默认值为1000
        :type num_classes: int
        :param fully_conv: 是否采用全卷积神经网络, 默认值为False
        :type fully_conv: bool
        :param remove_avg_pool_layer: 是否移除平均池化层, 默认值为False
        :type remove_avg_pool_layer: bool
        :param output_stride: resnet的最大步长, 默认值为32
        :type output_stride: int
        """        
        # Add additional variables to track
        # output stride. Necessary to achieve
        # specified output stride.
        self.output_stride = output_stride
        """整个ResNet网络的累积步长"""
        self.current_stride = 4
        """当前ResNet网络的累积步长,做内部标志用,动态变化"""
        self.current_dilation = 1
        """当前神经层的膨胀率,用来模拟步长操作,做内部标志用,动态变化"""
        self.remove_avg_pool_layer = remove_avg_pool_layer
        """是否移除最后的平均池化层(仅当self.fully_conv=true时生效)"""
        self.inplanes = 64
        """当前残差层输入特征的数量,做内部标志用,动态变化"""
        self.fully_conv = fully_conv
        """是否采用全卷积神经网络"""
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.fully_conv:  # 若采用全卷积神经网络,则resnet最后一层采用卷积层,并修改平均池化层
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)
            # In the latest unstable torch 4.0 the tensor.copy_
            # method was changed and doesn't work as it used to be
            self.fc = nn.Conv2d(512 * block.expansion, num_classes, 1)
        else:                # 若为普通的resnet网络,则最后一层仍为全连接层
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化resnet网络中卷积层和batchnorm层的可学习参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        _make_layer 将blocks个block类型的残差块串联起来,并将生成的残差层以序列容器的形式返回

        :param block: 残差块的类型:基础残差块(BasicBlock)和瓶颈残差块(BottleneckBlock)
        :type block: BasicBlock类或Bottleneck类
        :param planes: 输出的特征数数量
        :type planes: int
        :param blocks: 串联的残差块的个数
        :type blocks: int
        :param stride: 步长, 默认值为1
        :type stride: int
        :param dilation: 膨胀率, 默认值为1
        :type dilation: int
        :return: 残差层的序列容器
        :rtype: torch.nn.Sequential实例

        .. note:: - ``planes`` 参数代表期望的输出特征数量,而 ``实际的特征数量 = planes*block.expansion`` 与残差块的类型
                    ``block`` 有关.当block为基础残差块时, ``block.expansion=1`` ;当block为瓶颈残差块时, ``block.expansion=4``
                  - ``stride`` 参数在整个残差层中仅生效一次,即输入的特征大小 ``(h,w)`` 与输出的特征大小 ``(H,W)`` 
                    满足 ``(H,W)=(h,w)/stride``

        .. warning:: ``dilation`` 参数在本函数中实际并未被使用
        """
        downsample = None

        # 若步长不为1或输入特征数不等于输出特征数,则需要对残差块的输入数据进行下采样(downsample)操作,使其和输出数据的大小匹配,以便进行残差计算
        if stride != 1 or self.inplanes != planes * block.expansion:

            # Check if we already achieved desired output stride.
            if self.current_stride >= self.output_stride:

                # If so, replace subsampling with a dilation to preserve
                # current spatial resolution.
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:

                # If not, perform subsampling and update current
                # new output stride.
                self.current_stride = self.current_stride * stride

            # We don't dilate 1x1 convolution.
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # self.current_dilation = 1
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=self.current_dilation))
        self.inplanes = planes * block.expansion  # 更新输入特征数量
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.current_dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward resnet网络神经层的组织形式
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x2s = self.relu(x)
        x = self.maxpool(x2s)

        x4s = self.layer1(x)
        x8s = self.layer2(x4s)
        x16s = self.layer3(x8s)
        x32s = self.layer4(x16s)
        x = x32s

        if self.fully_conv:
            if not self.remove_avg_pool_layer:
                x =self.avgpool(x)
        else:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        xfc = self.fc(x)

        return x2s, x4s, x8s, x16s, x32s, xfc


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

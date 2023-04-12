"""
networks.pvnet模块
================

基于ResNet(残差网络),构建PVNet神经网络.PVNet由四个嵌套式的残差模块组成,其具体结构如下:

- 输入数据 ``N,3,H,W``

  - ResNet18初始卷积层 ``N,64,H/2,W/2`` :卷积层 ``(N,3,H,W)->(N,64,H/2,W/2)`` ,BN层,ReLU激活函数

    - ResNet18残差层一 ``N,64,H/4,W/4`` :最大池化层 ``(N,64,H/2,W/2)->(N,64,H/4,W/4)`` ,BasicBlock,BasicBlock

      - ResNet18残差层二 ``N,128,H/8,W/8`` :BasicBlock ``(N,64,H/4,W/4)->(N,128,H/8,W/8)`` ,BasicNlock
      - ResNet18残差层三 ``N,256,H/8,W/8`` :BasicBlock ``(N,128,H/8,W/8)->(N,256,H/8,W/8)`` ,BasicNlock
      - ResNet18残差层四 ``N,512,H/8,W/8`` :BasicBlock ``(N,256,H/8,W/8)->(N,512,H/8,W/8)`` ,BasicNlock
      - ResNet18全连接层 ``N,fcdim,H/8,W/8`` :卷积层 ``(N,512,H/8,W/8)->(N,fcdim,H/8,W/8)`` ,BN层,ReLU激活函数
      - 8s卷积层 ``N,s8dim,H/8,W/8`` :卷积层 ``(N,128+fcdim,H/8,W/8)->(N,s8dim,H/8,W/8)`` ,BN层,LeakRelU激活函数
      - 上采样层 ``N,s8dim,H/4,W/4``

    - 4s卷积层 ``N,s4dim,H/4,W/4`` :卷积层 ``(N,64+s8dim,H/4,W/4)->(N,s4dim,H/4,W/4)`` ,BN层,LeakRelU激活函数
    - 上采样层 ``N,s4dim,H/2,W/2``

  - 2s卷积块 ``N,s2dim,H/2,W/2`` :卷积层 ``(N,64+s4dim,H/2,W/2)->(N,s2dim,H/2,W/2)`` ,BN层,LeakRelU激活函数
  - 上采样层 ``N,s2dim,H,W``

- raw卷积块 ``N,seg_dim+ver_dim,H,W`` :卷积层 ``(N,3+s2dim,H,W)->(N,raw_dim,H,W)`` ,BN层,LeakReLU激活函数,
  卷积层 ``(N,raw_dim,H,W)->(N,seg_dim+ver_dim,H,W)``

.. note:: 构建残差变种神经网络的过程中,要特别注意神经层将数据维度的匹配关系.如上所示,着重描述了数据维度的变化过程.
"""
from .resnet18  import get_res_pvnet

_network_factory = {
    'res': get_res_pvnet
}
"""network工厂"""


def get_network(cfg):
    """
    get_network 获取PVNet网络模型实例

    :param cfg: 配置管理器
    :type cfg: CfgNode类
    :return: PVNet实例
    :rtype: torch.nn.Module
    """
    arch = cfg.network
    get_model = _network_factory[arch]
    network = get_model(cfg.heads['vote_dim'], cfg.heads['seg_dim'])
    return network

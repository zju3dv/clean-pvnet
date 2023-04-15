"""
recoder模块
===========

本模块主要实现了一个记录器类(Recoder),用来记录神经网络训练和验证时的相关信息.
"""
# 标准库
import os
from collections import deque, defaultdict
# 第三方库
import torch
from tensorboardX import SummaryWriter

class SmoothedValue(object):
    """
    SmoothedValue 记录所有输入的数据,但仅能对有限个最近输入的数据进行访问.
    (Track a series of values and provide access to smoothed values over a window or the global series average)

    :param window_size: 窗口大小, 默认值为20
    :type window_size: int
    """    
    def __init__(self, window_size=20):
        """
        __init__ 初始化函数

        :param window_size: 窗口大小, 默认值为20
        :type window_size: int
        """
        self.deque = deque(maxlen=window_size)
        """双向队列(记录最近输入的数据)"""
        self.total = 0.0
        """实例化以来记录的所有数据之和"""
        self.count = 0
        """实例化以来记录的所有数据个数"""

    def update(self, value):
        """
        update 记录新的数据

        :param value: 待记录数据
        :type value: float
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        """
        median 计算队列中数据的中位数

        :return: 中位数
        :rtype: float
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        avg 计算队列中数据的均值

        :return: 均值
        :rtype: float
        """
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        """
        global_avg 计算所有数据的均值(当前保存于队列中的,或着曾经记录过但已从队列中挤出的)

        :return: 整体均值
        :rtype: float
        """
        return self.total / self.count


class Recorder(object):
    """
    Recorder 记录神经网络训练和测试过程中的相关信息:损失状态,图像状态,数据加载时间,批处理时间等.
    这些信息可供程序格式化输出至屏幕,也会被保存在指定目录内供tensorboard调用.

    :param cfg: 配置管理系统
    :type cfg: CfgNode
    """
    def __init__(self, cfg):
        """
        __init__ 初始化函数

        :param cfg: 配置管理系统
        :type cfg: CfgNode
        """
        log_dir = cfg.record_dir
        """日志目录(保存日志信息的文件根目录)"""
        if not cfg.resume:
            os.system('rm -rf {}'.format(log_dir))
        self.writer = SummaryWriter(log_dir=log_dir)
        """tensorboard.SummaryWriter实例"""

        # scalars
        self.epoch = 0
        """周期/回合数"""
        self.step = 0
        """步数"""
        self.loss_stats = defaultdict(SmoothedValue)
        """损失变化信息"""
        self.batch_time = SmoothedValue()
        """批处理时间信息"""
        self.data_time = SmoothedValue()
        """数据加载时间信息"""

        # images
        self.image_stats = defaultdict(object)
        if 'process_'+cfg.task in globals():
            self.processor = globals()['process_'+cfg.task]
        else:
            self.processor = None

    def update_loss_stats(self, loss_dict):
        """
        update_loss_stats 更新损失状态

        :param loss_dict: 本次迭代的损失
        :type loss_dict: dict
        """
        for k, v in loss_dict.items():
            self.loss_stats[k].update(v.detach().cpu())

    def update_image_stats(self, image_stats):
        """
        update_image_stats 更新图片状态

        :param image_stats: 本次迭代的图像效果
        :type image_stats: dict
        """
        if self.processor is None:
            return
        image_stats = self.processor(image_stats)
        for k, v in image_stats.items():
            self.image_stats[k] = v.detach().cpu()

    def record(self, prefix, step=-1, loss_stats=None, image_stats=None):
        """
        record 将截至至当前迭代次数为止的损失状态记录在相应的文件内

        :param prefix: 模式前缀(train/val)
        :type prefix: str
        :param step: 迭代次数/步数, 默认值为-1
        :type step: int
        :param loss_stats: 损失信息, 默认值为None
        :type loss_stats: dict
        :param image_stats: 图像信息, 默认值为None
        :type image_stats: dict
        """
        pattern = prefix + '/{}'
        step = step if step >= 0 else self.step
        loss_stats = loss_stats if loss_stats else self.loss_stats

        # 写入近20次损失状态的均值
        for k, v in loss_stats.items():
            if isinstance(v, SmoothedValue):
                self.writer.add_scalar(pattern.format(k), v.median, step)
            else:
                self.writer.add_scalar(pattern.format(k), v, step)

        if self.processor is None:
            return
        # 写入当前的图片状态
        image_stats = self.processor(image_stats) if image_stats else self.image_stats
        for k, v in image_stats.items():
            self.writer.add_image(pattern.format(k), v, step)

    def state_dict(self):
        """
        state_dict 读取步数/迭代次数

        :return: {'step':self.step}
        :rtype: dict
        """
        scalar_dict = {}
        scalar_dict['step'] = self.step
        return scalar_dict

    def load_state_dict(self, scalar_dict):
        """
        load_state_dict 更新步数/迭代次数
        """
        self.step = scalar_dict['step']

    def __str__(self):
        """
        __str__ 字符串函数.将记录器记录的近20条数据格式化在一个字符串内,字符串内包含的信息如下:

        - 当前优化回合/周期epoch
        - 当前迭代的总次数/步数step
        - 近20次损失的均值
        - 近20次数据加载时间的均值
        - 近20次批处理时间的均值

        :return: 类的字符串
        :rtype: str
        """
        loss_state = []
        for k, v in self.loss_stats.items():
            loss_state.append('{}: {:.4f}'.format(k, v.avg))
        loss_state = '  '.join(loss_state)

        recording_state = '  '.join(['epoch: {}', 'step: {}', '{}', 'data: {:.4f}', 'batch: {:.4f}'])
        return recording_state.format(self.epoch, self.step, loss_state, self.data_time.avg, self.batch_time.avg)


def make_recorder(cfg):
    """
    make_recorder 生成一个记录器实例

    :param cfg: 配置管理系统
    :type cfg: CfgNode
    :return: 记录器实例
    :rtype: Recoder
    """
    return Recorder(cfg)


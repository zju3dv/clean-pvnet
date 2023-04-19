"""
trainer模块
===========

本模块主要实现了一个训练器类(Trainer),集成了网络训练和验证的相关功能.
"""
# 标准库 
import time
import datetime
# 第三方库
import tqdm
import torch
from torch.nn import DataParallel


class Trainer(object):
    """
    Trainer 实现对神经网络的训练(train)和验证(val)任务.

    :param network: 神经网络实例
    :type network: torch.nn.Module实例
    """
    def __init__(self, network):
        """
        __init__ 初始化函数

        :param network: 神经网络实例
        :type network: torch.nn.Module实例
        """
        # network = network.cuda()  # 将网络中的参数和缓存移动到GPU内进行计算
        # network = DataParallel(network)  # 实现多GPU并行计算
        self.network = network
        """神经网络实例"""

    def reduce_loss_stats(self, loss_stats):
        """
        reduce_loss_stats 将网络计算得到的损失降维(即取均值)

        :param loss_stats: 损失
        :type loss_stats: dict
        :return: 降维后的损失
        :rtype: dict
        """
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        """
        to_cuda 返回batch在cuda中的副本

        :param batch: 批数据
        :type batch: dict
        :return: 批数据在cuda中的副本
        :rtype: dict
        """
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        """
        train 启动一个训练回合,迭代整个训练集,优化网络参数.其中每次迭代都会记录网络的损失状态及时间参数(数据加载时间,批处理时间).
        在每20次迭代或着对训练集的最后一次迭代时,将着近20次记录信息的均值格式化输出至屏幕,并调用record保存相关信息至文件内.

        :param epoch: 当前优化回合/周期
        :type epoch: int
        :param data_loader: 数据集加载器
        :type data_loader: torch.utils.data.DataLoader
        :param optimizer: 优化器
        :type optimizer: torch.optim.Optimizer
        :param recorder: 记录器
        :type recorder: Recoder
        """
        max_iter = len(data_loader)
        """最大迭代次数(即批数量)"""
        self.network.train()  # 设置神经网络为训练模式(使BN层在训练过程更新自身参数)
        end = time.time()
        for iteration, batch in tqdm.tqdm(enumerate(data_loader)):
            data_time = time.time() - end
            """数据加载时间"""
            iteration = iteration + 1
            """本回合迭代次数"""
            recorder.step += 1  # 记录器总迭代次数+1

            # batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = self.network(batch)  # 计算本次迭代过程中的结果(output)和损失(loss)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()  # 将参数中保存的梯度清零
            loss.backward()  # 反向传播计算参数中的梯度
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)  # 将梯度截断至[-40,40]的范围内
            optimizer.step()  # 基于相关优化算法更新参数

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)  # 将损失降至一维
            recorder.update_loss_stats(loss_stats)  # 记录本次迭代的损失

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)  # 记录批处理时间
            recorder.data_time.update(data_time)  # 记录数据加载时间

            # 当每迭代20次或最后一次迭代时,格式化输出当前的训练状态,记录当前的损失状态至日志文件
            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)  # 预计训练的剩余时间
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']  # 取网络当前学习率
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0   # 自追踪节点(默认为程序开始)以来,GPU内存的最大占用量,单位:字节

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)  # eta: xx:xx:xx  epoch: int  step: int  vote_loss: .4f  seg_loss: .4f  loss: .4f  data: .4f(s)  batch: .4f(s)  lr: .6f  max_mem: .0f(MB)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')


    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        """
        val 启动一个验证回合,迭代整个验证集,观察网络性能.最后会将本回合内网络在验证集上的平均损失以及评估结果格式化输出至屏幕上,
        并将这些信息保存在指定文件内.

        :param epoch: 当前回合数/周期
        :type epoch: int
        :param data_loader: 验证集数据加载器
        :type data_loader: torch.utils.data.DataLoader
        :param evaluator: 评估器, 默认值为None
        :type evaluator: Evaluator
        :param recorder: 记录器, 默认值为None
        :type recorder: Recoder
        """
        self.network.eval()  # 设置网络为验证模式(停止更新BN层参数)
        torch.cuda.empty_cache()  # 释放GPU缓存
        val_loss_stats = {}
        """整个验证集的损失之和"""
        data_size = len(data_loader)
        """验证集大小"""
        # 迭代验证集
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()  # 生成对应tensor在cuda内存的副本

            with torch.no_grad():  # 不需要计算梯度(停止构建计算图)
                output, loss, loss_stats, image_stats = self.network.module(batch)
                # 评估网络的预测结果
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)  # 损失降至一维
            for k, v in loss_stats.items():  # 累加损失
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        # 格式化输出验证集的平均损失至屏幕
        loss_state = []
        """验证集的平均损失(格式化字符换)"""
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        # 格式化输出评估结果,并将评估结果保存在损失列表内(val_loss_stats)
        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        # 记录本回合损失状态和图像状态至指定文件内
        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)


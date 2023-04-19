"""
config子模块
============

基于yacs管理项目的配置信息，被管理的配置信息主要包括以下几个方面：

- 
- 
- 
- 
"""

# 标准库
import os
import sys
import argparse

#第三方库
from yacs.config import CfgNode as CN
import open3d

cfg = CN()

# model
cfg.model = 'custom'
cfg.model_dir = 'data/model'

# network heads
cfg.heads = ''

# task
cfg.task = 'pvnet'

# if load the pretrained network
cfg.resume = True

# epoch
cfg.ep_iter = -1
cfg.save_ep = 1
cfg.eval_ep = 5

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'CustomTrain'
cfg.train.epoch = 20
cfg.train.num_workers = 8
cfg.train.batch_size = 16

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-3
cfg.train.weight_decay = 0.

# scheduler
cfg.train.warmup = False
cfg.train.milestones = [20, 40, 60, 80, 100, 120, 160, 180, 200, 220]
cfg.train.gamma = 0.5

# augmentation
cfg.train.affine_rate = 0.
cfg.train.cropresize_rate = 1.0
cfg.train.rotate_rate = 1.0
cfg.train.rotate_min = -30
cfg.train.rotate_max = 30

cfg.train.overlap_ratio = 0.8
cfg.train.resize_ratio_min = 0.8
cfg.train.resize_ratio_max = 1.2

cfg.train.batch_sampler = 'image_size'

# -----------------------------------------------------------------------------
# val and test
# -----------------------------------------------------------------------------

# val
cfg.is_val = True
cfg.val = CN()
cfg.val.dataset = 'CustomVal'

# test
cfg.test = CN()
cfg.test.dataset = 'CustomTest'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.icp = False
cfg.test.un_pnp = False
cfg.test.vsd = False
cfg.test.det_gt = False

cfg.test.batch_sampler = 'image_size'

cfg.det_meta = CN()
cfg.det_meta.arch = 'dla'
cfg.det_meta.num_layers = 34
cfg.det_meta.heads = CN({'ct_hm': 1, 'wh': 2})

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

# dataset
cfg.cls_type = 'charger'

# tless
cfg.tless = CN()
cfg.tless.pvnet_input_scale = (256, 256)
cfg.tless.scale_train_ratio = (1.8, 2.4)
cfg.tless.scale_ratio = 2.4
cfg.tless.box_train_ratio = (1.0, 1.2)
cfg.tless.box_ratio = 1.2
cfg.tless.rot = 360.
cfg.tless.ratio = 0.8

_heads_factory = {
    'pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct_pvnet': CN({'vote_dim': 18, 'seg_dim': 2}),
    'ct': CN({'ct_hm': 30, 'wh': 2})
}


def parse_cfg(cfg, args):
    """
    parse_cfg _summary_

    :param cfg: _description_
    :type cfg: _type_
    :param args: _description_
    :type args: _type_
    :raises ValueError: _description_
    """
    # 任务类型必须被指定
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # # assign the gpus(单机分布式多卡训练时配置可使用显卡及顺序)
    # os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    # 配置networks heads
    if cfg.task in _heads_factory:
        cfg.heads = _heads_factory[cfg.task]

    # if 'Tless' in cfg.test.dataset and cfg.task == 'pvnet':
    #     cfg.cls_type = '{:02}'.format(int(cfg.cls_type))

    # if 'Ycb' in cfg.test.dataset and cfg.task == 'pvnet':
    #     cfg.cls_type = '{}'.format(int(cfg.cls_type))

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # 配置模型、记录、结果路径
    cfg.model_dir = os.path.join(cfg.model_dir, cfg.task, cfg.model)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def make_cfg(args):
    """
    make_cfg _summary_

    :param args: _description_
    :type args: _type_
    :return: _description_
    :rtype: _type_
    """
    # 将命令行中尚未匹配的参数提取至opts，并剔除其中错误参数
    opts_idx = [i for i in range(0, len(args.opts), 2) if args.opts[i].split('.')[0] in cfg.keys()]
    opts = sum([[args.opts[i], args.opts[i + 1]] for i in opts_idx], [])
    # 基于opts更新配置cfg
    cfg.merge_from_list(opts)
    parse_cfg(cfg, args)
    return cfg

# 兼容shpinx用
PATH_SPHINX = "/home/administrator/anaconda3/envs/pvnet/bin/sphinx-build"

if sys.argv[0] == PATH_SPHINX:
    args = argparse.Namespace()
else:
    # 利用argparse对命令行参数进行解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', dest='test', default=False)      # "--test" 
    parser.add_argument("--type", type=str, default="")                                 # "--type" 指定要训练的数据集(linemod, custom, etc.)或要进行的任务(visualize, evaluate, etc.)
    parser.add_argument('--det', type=str, default='')                                  # "--det" 
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)                 # "--opts"
    args = parser.parse_args()
    if len(args.type) > 0:
        cfg.task = "run"
    cfg = make_cfg(args)
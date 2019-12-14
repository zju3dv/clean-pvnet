from lib.config import cfg, args
import numpy as np
import os


def run_rgb():
    import glob
    from scipy.misc import imread
    import matplotlib.pyplot as plt

    syn_ids = sorted(os.listdir('data/ShapeNet/renders/02958343/'))[-10:]
    for syn_id in syn_ids:
        pkl_paths = glob.glob('data/ShapeNet/renders/02958343/{}/*.pkl'.format(syn_id))
        np.random.shuffle(pkl_paths)
        for pkl_path in pkl_paths:
            img_path = pkl_path.replace('_RT.pkl', '.png')
            img = imread(img_path)
            plt.imshow(img)
            plt.show()


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch['inp'])
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize(output, batch)


def run_visualize_train():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.visualizers import make_visualizer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=True)
    visualizer = make_visualizer(cfg, 'train')
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        visualizer.visualize_train(output, batch)


def run_analyze():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    from lib.analyzers import make_analyzer

    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    analyzer = make_analyzer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        analyzer.analyze(output, batch)


def run_net_utils():
    from lib.utils import net_utils
    import torch
    import os

    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    pretrained_model = torch.load(model_path)
    net = pretrained_model['net']
    net = net_utils.remove_net_prefix(net, 'dla.')
    net = net_utils.remove_net_prefix(net, 'cp.')
    pretrained_model['net'] = net
    model_path = 'data/model/rcnn_snake/rcnn/139.pth'
    os.system('mkdir -p {}'.format(os.path.dirname(model_path)))
    torch.save(pretrained_model, model_path)


def run_linemod():
    from lib.datasets.linemod import linemod_to_coco
    linemod_to_coco.linemod_to_coco(cfg)


def run_tless():
    from lib.datasets.tless import handle_rendering_data, fuse, handle_test_data, handle_ag_data, tless_to_coco
    # handle_rendering_data.render()
    # handle_rendering_data.render_to_coco()
    # handle_rendering_data.prepare_asset()

    # fuse.fuse()
    # handle_test_data.get_mask()
    # handle_test_data.test_to_coco()
    handle_test_data.test_pose_to_coco()

    # handle_ag_data.ag_to_coco()
    # handle_ag_data.get_ag_mask()
    # handle_ag_data.prepare_asset()

    # tless_to_coco.tless_train_to_coco()


def run_custom():
    from tools import handle_custom_dataset
    data_root = 'data/custom'
    handle_custom_dataset.sample_fps_points(data_root)
    handle_custom_dataset.custom_to_coco(data_root)


if __name__ == '__main__':
    globals()['run_'+args.type]()


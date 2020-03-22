from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        plt.imshow(inp)
        plt.plot(kpt_2d[:, 0], kpt_2d[:, 1], '.')
        plt.show()

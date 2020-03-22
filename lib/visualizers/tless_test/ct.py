from lib.utils import img_utils, data_utils
from lib.utils.tless import tless_config
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

mean = tless_config.mean
std = tless_config.std


class Visualizer:
    def __init__(self, split):
        self.split = split

    def visualize(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        box = output['detection'][0, :, :4].detach().cpu().numpy() * tless_config.down_ratio
        score = output['detection'][0, :, 4].detach().cpu().numpy()
        plt.imshow(inp)
        for i in range(len(box)):
            x_min, y_min, x_max, y_max = box[i]
            plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])
        plt.show()


import torch.nn as nn
from lib.networks.ct.dla import get_network as get_pose_net
from lib.networks.pvnet.resnet18 import get_res_pvnet
from lib.utils import net_utils, data_utils
from lib.utils.tless import tless_config, tless_pvnet_utils
import torch
import numpy as np
import cv2
from lib.config import cfg
from yacs.config import CfgNode as CN
fx_config = CN()


def _crop(img, box, trans_output_inv, output):
    box = data_utils.affine_transform(box.reshape(-1, 2), trans_output_inv).ravel()
    center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    scale = max(box[2] - box[0], box[3] - box[1]) * tless_config.scale_ratio

    input_h, input_w = tless_pvnet_utils.input_scale
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])

    img = img.astype(np.uint8).copy()
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - tless_config.mean) / tless_config.std
    inp = inp.transpose(2, 0, 1)
    inp = torch.Tensor(inp).cuda().float()[None]

    init = [inp, center, scale]

    return init


def crop(img, detection, batch, output):
    img = img[0].detach().cpu().numpy()
    fx_config.max_det = 1
    fx_config.ct_score = 0
    fx_config.down_ratio = 4

    box = output['detection'][0, :fx_config.max_det, :4]
    score = output['detection'][0, :fx_config.max_det, 4]
    box = box[score > fx_config.ct_score]
    box = box.detach().cpu().numpy() * fx_config.down_ratio

    center = batch['meta']['center'][0].detach().cpu().numpy()
    scale = batch['meta']['scale'][0].detach().cpu().numpy()
    h, w = batch['inp'].size(2), batch['inp'].size(3)
    trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)

    init = [_crop(img, box_, trans_output_inv, output) for box_ in box]
    if len(init) == 0:
        output.update({'inp': [], 'center': [], 'scale': []})
        return []

    inp, center, scale = list(zip(*init))
    inp = torch.cat(inp, dim=0)
    output.update({'inp': inp, 'center': center, 'scale': scale})

    return inp


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        det_meta = cfg.det_meta
        self.dla_ct = get_pose_net(det_meta.num_layers, det_meta.heads)
        self.pvnet = get_res_pvnet(cfg.heads['vote_dim'], cfg.heads['seg_dim'])

        net_utils.load_network(self.dla_ct, cfg.det_model)
        net_utils.load_network(self.pvnet, cfg.kpt_model)

    def forward(self, x, batch=None, id=0):
        output = self.dla_ct(x)

        inp = crop(batch['img'], output['detection'], batch, output)
        if len(inp) == 0:
            return output

        kpt_output = self.pvnet(inp)
        output.update({'kpt_2d': kpt_output['kpt_2d'], 'mask': kpt_output['mask']})

        return output


def get_network():
    network = Network()
    return network

from lib.utils.tless import tless_config, tless_pvnet_utils
from lib.utils import data_utils
import numpy as np
import cv2
from .tless_utils import *

_data_rng = tless_config.data_rng
_eig_val = tless_config.eig_val
_eig_vec = tless_config.eig_vec
mean = tless_config.mean
std = tless_config.std


def augment(img, split):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(img.shape[0], img.shape[1]) * 1.0
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    if split != 'train':
        center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        scale = max(width, height) * 1.0
        scale = np.array([scale, scale])
        x = 32
        input_w, input_h = int((width / 1. + x - 1) // x * x), int((height / 1. + x - 1) // x * x)

    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // tless_config.down_ratio, input_w // tless_config.down_ratio
    trans_output = data_utils.get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, trans_input, trans_output, center, scale, inp_out_hw


def pvnet_transform(img, box):
    center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.], dtype=np.float32)
    scale = np.array([box[2] - box[0], box[3] - box[1]], dtype=np.float32) * 1.2

    input_w, input_h = tless_pvnet_utils.input_scale
    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    return orig_img, inp, center, scale

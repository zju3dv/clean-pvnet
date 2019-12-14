import numpy as np
from lib.utils.pvnet import pvnet_config


mean = pvnet_config.mean
std = pvnet_config.std

def augment(img, split):
    orig_img = img.copy()
    inp = img

    inp = (inp.astype(np.float32) / 255.)
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1).astype(np.float32)

    return orig_img, inp



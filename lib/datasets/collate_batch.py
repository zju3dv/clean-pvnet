from torch.utils.data.dataloader import default_collate
import torch
import numpy as np


_collators = {
}


def make_collator(cfg):
    if cfg.task in _collators:
        return _collators[cfg.task]
    else:
        return default_collate

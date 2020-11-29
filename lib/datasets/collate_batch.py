from torch.utils.data.dataloader import default_collate
import torch
import numpy as np


def ct_collator(batch):
    ret = {'inp': default_collate([b['inp'] for b in batch])}
    ret.update({'img': default_collate([b['img'] for b in batch])})

    meta = default_collate([b['meta'] for b in batch])
    ret.update({'meta': meta})

    # detection
    ct_hm = default_collate([b['ct_hm'] for b in batch])

    batch_size = len(batch)
    ct_num = torch.max(meta['ct_num'])
    wh = torch.zeros([batch_size, ct_num, 2], dtype=torch.float)
    ct_cls = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_ind = torch.zeros([batch_size, ct_num], dtype=torch.int64)
    ct_01 = torch.zeros([batch_size, ct_num], dtype=torch.uint8)
    for i in range(batch_size):
        ct_01[i, :meta['ct_num'][i]] = 1

    wh[ct_01] = torch.Tensor(sum([b['wh'] for b in batch], []))
    ct_cls[ct_01] = torch.LongTensor(sum([b['ct_cls'] for b in batch], []))
    ct_ind[ct_01] = torch.LongTensor(sum([b['ct_ind'] for b in batch], []))

    detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'ct_01': ct_01.float()}
    ret.update(detection)

    return ret


def pvnet_collator(batch):
    if 'pose_test' not in batch[0]['meta']:
        return default_collate(batch)

    inp = np.stack(batch[0]['inp'])
    inp = torch.from_numpy(inp)
    meta = default_collate([b['meta'] for b in batch])
    ret = {'inp': inp, 'meta': meta}

    return ret


_collators = {
    'ct': ct_collator,
    'pvnet': pvnet_collator
}


def make_collator(cfg):
    if cfg.task in _collators:
        return _collators[cfg.task]
    else:
        return default_collate

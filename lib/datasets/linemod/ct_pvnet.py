import torch.utils.data as data
import cv2
import numpy as np
import math
from lib.utils import data_utils
from pycocotools.coco import COCO
import os
from lib.utils.tless import tless_test_utils, visualize_utils, tless_config


class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split, transforms=None):
        '''
        This dataset is only used for visualization.
        '''

        super(Dataset, self).__init__()

        self.split = split

        self.coco = COCO(ann_file)
        self.anns = np.array(self.coco.getImgIds())
        self.anns = self.anns[::10] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        img = cv2.imread(path)

        return img

    def __getitem__(self, index_tuple):
        index, height, width = index_tuple
        img_id = self.anns[index]

        img = self.read_data(img_id)
        orig_img, inp, trans_input, trans_output, center, scale, inp_out_hw = \
                tless_test_utils.augment(img, self.split)

        meta = {'center': center, 'scale': scale}
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret = {'inp': inp, 'img': img_rgb, 'img_id': img_id}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)


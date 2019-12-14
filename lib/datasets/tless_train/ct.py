import torch.utils.data as data
import cv2
import numpy as np
import math
from lib.utils import data_utils
from pycocotools.coco import COCO
import os
from lib.utils.tless import visualize_utils, tless_config
from PIL import Image
import glob
from lib.utils.tless import tless_train_utils
import time


class Dataset(data.Dataset):
    def __init__(self, ann_file, split):
        super(Dataset, self).__init__()

        self.split = split

        self.coco = COCO(ann_file)
        self.anns = np.array(self.coco.getImgIds())
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

        self.bg_paths = np.array(glob.glob('data/sun/JPEGImages/*.jpg'))

    def read_original_data(self, index):
        np.random.seed((int(time.time()) * (index + 1)) % (2 ** 16))
        img_ids = np.random.choice(self.anns, tless_config.num_obj_in_training_image)

        rgb_paths = []
        mask_paths = []
        category_ids = []
        for instance_id, img_id in enumerate(img_ids):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anno = self.coco.loadAnns(ann_ids)[0]
            rgb_paths.append(self.coco.loadImgs(int(img_id))[0]['rgb_path'])
            mask_paths.append(anno['mask_path'])
            category_ids.append(anno['category_id'])

        cls_ids = [self.json_category_id_to_contiguous_id[category_id] for category_id in category_ids]

        return rgb_paths, mask_paths, category_ids, cls_ids

    def prepare_detection(self, box, ct_hm, cls_id, wh, ct_cls, ct_ind):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct = np.round(ct).astype(np.int32)

        h, w = y_max - y_min, x_max - x_min
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)

        wh.append([w, h])
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])

        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]

        return decode_box

    def __getitem__(self, index):
        rgb_paths, mask_paths, category_ids, cls_ids = self.read_original_data(index)
        img, mask = tless_train_utils.cut_and_paste(self.bg_paths, rgb_paths, mask_paths, category_ids)

        bboxes = tless_train_utils.get_bbox(mask, category_ids)
        orig_img, inp, trans_input, trans_output, center, scale, inp_out_hw = \
                tless_train_utils.augment(img, bboxes, self.split)
        mask = cv2.warpAffine(mask, trans_output, (inp_out_hw[3], inp_out_hw[2]), flags=cv2.INTER_NEAREST)
        bboxes = tless_train_utils.get_bbox(mask, category_ids)
        # visualize_utils.visualize_bbox(orig_img, np.array(bboxes) * tless_config.down_ratio)

        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([30, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []

        bboxes_ = []
        for i in range(len(bboxes)):
            cls_id = cls_ids[i]
            bbox = bboxes[i]
            if len(bbox) == 0:
                continue
            if bbox[2] == bbox[0] or bbox[3] == bbox[1]:
                continue
            bboxes_.append(bbox)

            self.prepare_detection(bbox, ct_hm, cls_id, wh, ct_cls, ct_ind)

        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        ret.update(detection)
        # visualize_utils.visualize_detection(orig_img, ret)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)


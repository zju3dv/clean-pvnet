import torch.utils.data as data
import cv2
import numpy as np
import math
from lib.utils import data_utils
from pycocotools.coco import COCO
import os
from lib.utils.tless import tless_test_utils, visualize_utils, tless_config


class Dataset(data.Dataset):
    def __init__(self, ann_file, split):
        super(Dataset, self).__init__()

        self.split = split

        self.coco = COCO(ann_file)
        self.anns = np.array(self.coco.getImgIds())
        self.anns = self.anns[::10] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(int(img_id))[0]['rgb_path']
        img = cv2.imread(path)
        bboxes = [tless_test_utils.xywh_to_xyxy(obj['bbox']) for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]

        return img, bboxes, cls_ids

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
        img_id = self.anns[index]

        img, bboxes, cls_ids = self.read_data(img_id)
        orig_img, inp, trans_input, trans_output, center, scale, inp_out_hw = \
                tless_test_utils.augment(img, self.split)
        bboxes = tless_test_utils.transform_bbox(bboxes, trans_output, inp_out_hw[2], inp_out_hw[3])

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
            bboxes_.append(bbox)

            self.prepare_detection(bbox, ct_hm, cls_id, wh, ct_cls, ct_ind)

        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'ct_cls': ct_cls, 'ct_ind': ct_ind}
        ret.update(detection)
        # visualize_utils.visualize_bbox(orig_img, np.array(bboxes_) * tless_config.down_ratio)

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)


import torch.utils.data as data
import cv2
import numpy as np
import math
from lib.utils import data_utils
from pycocotools.coco import COCO
import os
from lib.utils.tless import tless_utils, visualize_utils, tless_config
from PIL import Image
import glob


class Dataset(data.Dataset):
    def __init__(self, ann_file, split):
        super(Dataset, self).__init__()

        self.split = split

        self.coco = COCO(ann_file)
        self.anns = np.array(self.coco.getImgIds())
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}

        self.bg_paths = np.array(glob.glob('data/sun/JPEGImages/*.jpg'))

    def get_training_data(self, index):
        np.random.seed(index)
        img_ids = np.random.choice(self.anns, tless_config.num_obj_in_training_image)
        train_img = cv2.imread(self.bg_paths[np.random.randint(len(self.bg_paths))])
        train_img = cv2.resize(train_img, (tless_config.train_w, tless_config.train_h))
        train_mask = np.zeros((tless_config.train_h, tless_config.train_w), dtype=np.int16)

        rgb_paths = []
        mask_paths = []
        category_ids = []
        for instance_id, img_id in enumerate(img_ids):
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anno = self.coco.loadAnns(ann_ids)[0]
            rgb_paths.append(self.coco.loadImgs(int(img_id))[0]['rgb_path'])
            mask_paths.append(anno['mask_path'])
            category_ids.append(anno['category_id'])

        for instance_id in range(len(rgb_paths)):
            rgb_path = rgb_paths[instance_id]
            mask_path = mask_paths[instance_id]
            category_id = category_ids[instance_id]

            img = cv2.imread(rgb_path)
            mask = np.array(Image.open(mask_path))

            mask_id = category_id * 1000 + instance_id
            tless_utils.cut_and_paste(img, mask, train_img, train_mask, mask_id)

        cls_ids = [self.json_category_id_to_contiguous_id[category_id] for category_id in category_ids]

        return train_img, train_mask, category_ids, cls_ids

    def get_bbox(self, mask, category_ids, trans_output, out_h, out_w):
        bboxes = []
        for instance_id in range(len(category_ids)):
            category_id = category_ids[instance_id]
            mask_id = category_id * 1000 + instance_id
            mask_ = (mask == mask_id).astype(np.float32)
            mask_ = cv2.warpAffine(mask_, trans_output, (out_w, out_h), flags=cv2.INTER_LINEAR)
            mask_ = (mask_ != 0).astype(np.uint8)
            bbox = tless_utils.xywh_to_xyxy(cv2.boundingRect(mask_))
            bbox[2] = min(bbox[2], out_w-1)
            bbox[3] = min(bbox[3], out_h-1)
            bboxes.append(bbox)
        return bboxes

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
        img, train_mask, category_ids, cls_ids = self.get_training_data(index)

        orig_img, inp, trans_input, trans_output, center, scale, inp_out_hw = \
                tless_utils.augment(img, self.split)
        bboxes = self.get_bbox(train_mask, category_ids, trans_output, inp_out_hw[2], inp_out_hw[3])

        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([30, output_h, output_w], dtype=np.float32)
        wh = []
        ct_cls = []
        ct_ind = []

        bboxes_ = []
        for i in range(len(bboxes)):
            cls_id = cls_ids[i]
            bbox = bboxes[i]
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


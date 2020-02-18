import torch.utils.data as data
import cv2
import numpy as np
import math
from lib.utils import data_utils
from pycocotools.coco import COCO
import os
from lib.utils.tless import tless_test_utils, visualize_utils, tless_config, tless_pvnet_utils
import json
from lib.config import cfg


class CocoDet:
    def __init__(self, ann_file):
        super(CocoDet, self).__init__()

        self.dets = json.load(open(ann_file, 'r'))
        self.imgId_to_dets = {}
        self.objId_to_imgIds = {}
        for det in self.dets:
            self.imgId_to_dets.setdefault(det['image_id'], []).append(det)
            self.objId_to_imgIds.setdefault(det['category_id'], set()).add(det['image_id'])

    def getImgIds(self, obj_id):
        return list(self.objId_to_imgIds[obj_id])

    def getBBoxes(self, img_id, obj_id):
        dets = [det for det in self.imgId_to_dets[img_id] if det['category_id'] == obj_id]
        bboxes = [det['bbox'] for det in dets]
        return bboxes


class Dataset(data.Dataset):
    def __init__(self, det_file, ann_file, det_gt_file, obj_id, split):
        super(Dataset, self).__init__()

        self.split = split

        if cfg.test.det_gt:
            self.coco_det_gt = COCO(det_gt_file)
            self.path_to_img_id = {}
            for img_data in self.coco_det_gt.dataset['images']:
                self.path_to_img_id[img_data['rgb_path']] = img_data['id']

        self.coco_det = CocoDet(det_file)
        self.coco = COCO(ann_file)
        self.obj_id = int(obj_id)

        gt_img_ids = self.coco.getImgIds(catIds=[self.obj_id])
        det_img_ids = self.coco_det.getImgIds(self.obj_id)

        if cfg.test.det_gt:
            self.anns = np.array(gt_img_ids)
        else:
            self.anns = np.array(list(set(gt_img_ids).intersection(det_img_ids)))

        self.anns = self.anns[::3] if split == 'mini' else self.anns

    def read_data(self, img_id):
        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        img = cv2.imread(path)

        if cfg.test.det_gt:
            det_gt_id = self.path_to_img_id[path]
            ann_ids = self.coco_det_gt.getAnnIds(imgIds=det_gt_id)
            anno = self.coco_det_gt.loadAnns(ann_ids)
            anno = [a for a in anno if a['category_id'] == self.obj_id]
            bboxes = [a['bbox'] for a in anno]
        else:
            bboxes = self.coco_det.getBBoxes(img_id, self.obj_id)
        bboxes = [tless_test_utils.xywh_to_xyxy(box) for box in bboxes]

        return img, bboxes

    def __getitem__(self, index):
        img_id = self.anns[index]

        img, bboxes = self.read_data(img_id)
        data_list = [tless_test_utils.pvnet_transform(img, box) for box in bboxes]
        orig_imgs, inps, centers, scales = [list(d) for d in zip(*data_list)]
        inp_hw = tless_pvnet_utils.input_scale.tolist()

        ret = {'inp': inps}
        bboxes = [np.array(box) for box in bboxes]
        meta = {'center': centers, 'scale': scales, 'box': bboxes, 'img_id': img_id, 'pose_test': ''}
        ret.update({'meta': meta})
        # visualize_utils.visualize_img(orig_imgs)

        return ret

    def __len__(self):
        return len(self.anns)


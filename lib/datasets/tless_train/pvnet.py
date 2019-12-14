import torch.utils.data as data
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
from lib.utils.pvnet import pvnet_data_utils, pvnet_linemod_utils, visualize_utils
from lib.utils.linemod import linemod_config
from lib.datasets.augmentation import crop_or_padding_to_fixed_size, rotate_instance, crop_resize_instance_v1
import random
import torch
import glob
from lib.config import cfg
from lib.utils.tless import tless_train_utils, tless_utils, tless_config, tless_pvnet_utils
from lib.utils import data_utils
import cv2


class Dataset(data.Dataset):

    def __init__(self, ann_file, split):
        super(Dataset, self).__init__()

        self.split = split

        self.coco = COCO(ann_file)
        self.img_ids = np.array(sorted(self.coco.getImgIds()))

        self.bg_paths = np.array(glob.glob('data/sun/JPEGImages/*.jpg'))

        self.other_obj_coco = COCO('data/tless/train_primesense/assets/train.json')
        cat_ids = self.other_obj_coco.getCatIds()
        cat_ids.remove(int(cfg.cls_type))
        self.oann_ids = self.other_obj_coco.getAnnIds(catIds=cat_ids)

    def read_data(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anno = self.coco.loadAnns(ann_ids)[0]

        path = self.coco.loadImgs(int(img_id))[0]['file_name']
        inp = cv2.imread(path)
        kpt_2d = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)
        mask = pvnet_data_utils.read_tless_mask(anno['mask_path'])

        rot = np.random.uniform() * 360
        inp, _ = tless_train_utils.rotate_image(inp, rot, get_rot=True)
        if np.random.uniform() < 0.8:
            inp = tless_train_utils.color_jitter.augment_image(inp)
        mask, rot = tless_train_utils.rotate_image(mask, rot, get_rot=True)
        kpt_2d = data_utils.affine_transform(kpt_2d, rot)

        return inp, kpt_2d, mask

    def get_training_img(self, img, kpt_2d, mask):
        pixel_num = np.sum(mask)

        train_img = cv2.imread(self.bg_paths[np.random.randint(len(self.bg_paths))])
        train_img = cv2.resize(train_img, (tless_config.train_w, tless_config.train_h))
        train_mask = np.zeros((tless_config.train_h, tless_config.train_w), dtype=np.uint8)
        tless_utils.cut_and_paste(img, mask, train_img, train_mask, 1)
        x_min, y_min, _, _ = cv2.boundingRect(mask)
        x, y, w, h = cv2.boundingRect(train_mask)
        kpt_2d = kpt_2d - [x_min, y_min] + [x, y]

        fused_img, fused_mask = tless_train_utils.get_fused_image(self.other_obj_coco, self.oann_ids, self.bg_paths)

        def paste_img0_on_img1(img0, mask0, img1, mask1):
            img, mask = img1.copy(), mask1.copy()
            mask_ = mask0 == 1
            img[mask_] = img0[mask_]
            mask[mask_] = 0
            return img, mask

        if np.random.uniform() < 0.5:
            train_img, _ = paste_img0_on_img1(train_img, train_mask, fused_img, fused_mask)
        else:
            img, mask = paste_img0_on_img1(fused_img, fused_mask, train_img, train_mask)
            if np.sum(mask) / pixel_num < 0.2:
                train_img, _ = paste_img0_on_img1(train_img, train_mask, fused_img, fused_mask)
            else:
                train_img, train_mask = img, mask

        x, y, w, h = cv2.boundingRect(train_mask)
        bbox = [x, y, x + w - 1, y + h - 1]

        return train_img, kpt_2d, train_mask, bbox

    def __getitem__(self, index):
        index = 0
        img_id = self.img_ids[index]
        img, kpt_2d, mask = self.read_data(img_id)
        img, kpt_2d, mask, bbox = self.get_training_img(img, kpt_2d, mask)

        orig_img, inp, trans_input, center, scale, inp_hw = \
            tless_pvnet_utils.augment(img, bbox, 'train')
        kpt_2d = data_utils.affine_transform(kpt_2d, trans_input)
        mask = cv2.warpAffine(mask, trans_input, (inp_hw[1], inp_hw[0]), flags=cv2.INTER_NEAREST)
        vertex = pvnet_data_utils.compute_vertex(mask, kpt_2d).transpose(2, 0, 1)

        ret = {'inp': inp, 'mask': mask.astype(np.uint8), 'vertex': vertex}
        # visualize_utils.visualize_ann(orig_img, kpt_2d, mask, False)

        return ret

    def __len__(self):
        return len(self.img_ids)

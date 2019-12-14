import numpy as np
import cv2
import os
import glob
from PIL import Image
import tqdm
from lib.utils import base_utils


num_train_imgs = 80000
max_objects_in_scene = 6

bg_dir = '/mnt/data/home/pengsida/Datasets/SUN2012pascalformat/JPEGImages/*.jpg'
bg_paths = glob.glob(bg_dir)
output_dir = '/mnt/data/home/pengsida/Datasets/tless/tless-mix'
output_rgb_dir = os.path.join(output_dir, 'rgb')
output_mask_dir = os.path.join(output_dir, 'mask')
tless_dir = '/home/pengsida/Datasets/tless/renders'


def cut_and_paste(img, mask, train_img, train_mask, instance_id):
    ys, xs = np.nonzero(mask)
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    h, w = y_max - y_min, x_max - x_min
    img_h, img_w = train_img.shape[0], train_img.shape[1]

    dst_y, dst_x = np.random.randint(0, img_h-h), np.random.randint(0, img_w-w)
    dst_ys, dst_xs = ys - y_min + dst_y, xs - x_min + dst_x

    train_img[dst_ys, dst_xs] = img[ys, xs]
    train_mask[dst_ys, dst_xs] = instance_id


def fuse():
    W, H = 720, 540
    noofobjects = 30

    if not os.path.exists(output_rgb_dir):
        os.makedirs(output_rgb_dir)
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)

    noofimages = {i+1: len(glob.glob(os.path.join(tless_dir, str(i+1), '*.pkl'))) for i in range(noofobjects)}

    obj_info = []

    for i in tqdm.tqdm(range(num_train_imgs)):
        train_img = np.zeros((H, W, 3), dtype=np.uint8)
        train_mask = np.zeros((H, W), dtype=np.uint8)
        instance_id = 0

        ann = []
        for k in range(max_objects_in_scene):
            obj_id = np.random.randint(0, noofobjects) + 1
            img_id = np.random.randint(0, noofimages[obj_id])
            img_path = os.path.join(tless_dir, str(obj_id), str(img_id)+'.png')
            dpt_path = os.path.join(tless_dir, str(obj_id), str(img_id)+'_depth.png')

            rand_img = cv2.imread(img_path)
            dpt = np.array(Image.open(dpt_path))
            mask = (dpt != 65535).astype(np.uint8)

            instance_id += 1
            cut_and_paste(rand_img, mask, train_img, train_mask, instance_id)
            ann.append([obj_id, instance_id])

        new_img = cv2.resize(cv2.imread(bg_paths[np.random.randint(0, len(bg_paths))]), (W, H))
        fg_mask = train_mask != 0
        new_img[fg_mask] = train_img[fg_mask]

        img_path = os.path.join(output_rgb_dir, str(i)+'.png')
        mask_path = os.path.join(output_mask_dir, str(i)+'.png')
        obj_info.append({'img_path': img_path, 'mask_path': mask_path, 'ann': ann})
        cv2.imwrite(img_path, new_img)
        cv2.imwrite(mask_path, train_mask)

    base_utils.save_pickle(obj_info, os.path.join(output_dir, 'obj_info.pkl'))

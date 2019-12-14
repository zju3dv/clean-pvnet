import os
import json
import tqdm
import yaml
from lib.utils import base_utils
import numpy as np
from lib.datasets.tless import opengl_renderer
from PIL import Image
import glob
import cv2


def ag_to_coco():
    data_root = 'data/tless/t-less-mix'
    ag_anns = os.path.join(data_root, 'annotations.csv')

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    hash_img_id = {}

    with open(ag_anns, 'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            line = line[:-1].split(',')
            path = line[0]
            x_min, y_min, x_max, y_max = [int(x) for x in line[1:5]]
            obj_id = int(line[5])

            img_id_str = os.path.basename(path)
            if img_id_str not in hash_img_id:
                hash_img_id[img_id_str] = 0
                rgb_path = os.path.join(data_root, 'rgb', img_id_str)
                img_id += 1
                info = {'rgb_path': rgb_path, 'id': img_id}
                images.append(info)

            ann_id += 1
            anno = {'image_id': img_id, 'bbox': [x_min, y_min, x_max, y_max], 'category_id': obj_id, 'id': ann_id}
            annotations.append(anno)

    obj_ids = [i + 1 for i in range(30)]
    categories = [{'supercategory': 'none', 'id': obj_id, 'name': str(obj_id)} for obj_id in obj_ids]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}
    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)


def get_rendering_model(data_root):
    rendering_model_path = os.path.join(data_root, 'rendering_model.pkl')
    if os.path.exists(rendering_model_path):
        return base_utils.read_pickle(rendering_model_path)

    cad_path_pattern = os.path.join(data_root, 'obj_{:03}.ply')
    obj_ids = [i + 1 for i in range(30)]
    models = {}
    for obj_id in tqdm.tqdm(obj_ids):
        cad_path = cad_path_pattern.format(obj_id)
        model = opengl_renderer.load_ply(cad_path)
        models.update({obj_id: model})
    base_utils.save_pickle(models, rendering_model_path)

    return models


def get_ag_mask():
    data_root = 'data/tless/train_primesense'
    obj_ids = [i + 1 for i in range(30)]
    models = get_rendering_model('data/tless/models_cad')

    for obj_id in tqdm.tqdm(obj_ids):
        obj_dir = os.path.join(data_root, '{:02}'.format(obj_id))
        rgb_dir = os.path.join(obj_dir, 'rgb')
        mask_dir = os.path.join(obj_dir, 'mask')
        os.system('mkdir -p {}'.format(mask_dir))

        gt = yaml.load(open(os.path.join(obj_dir, 'gt.yml'), 'r'))
        K_info = yaml.load(open(os.path.join(obj_dir, 'info.yml'), 'r'))
        img_ids = list(gt.keys())

        for img_id in tqdm.tqdm(img_ids):
            instance_gt = gt[img_id][0]
            K = K_info[img_id]['cam_K']
            K = np.array(K).reshape(3, 3)

            R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
            t = np.array(instance_gt['cam_t_m2c']) * 0.001
            pose = np.concatenate([R, t[:, None]], axis=1)

            rgb_path = os.path.join(rgb_dir, '{:04}.png'.format(img_id))
            w, h = Image.open(rgb_path).size

            obj_id = instance_gt['obj_id']
            depth = opengl_renderer.render(models[obj_id], pose, K, w, h)
            mask = (depth != 0).astype(np.uint8)

            mask_path = rgb_path.replace('rgb', 'mask')
            Image.fromarray(mask).save(mask_path, 'PNG')


def ag_to_asset():
    data_root = 'data/tless/train_primesense'
    obj_ids = [i + 1 for i in range(30)]
    asset_dir = os.path.join(data_root, 'assets')

    for obj_id in tqdm.tqdm(obj_ids):
        obj_dir = os.path.join(data_root, '{:02}'.format(obj_id))
        rgb_dir = os.path.join(obj_dir, 'rgb')
        mask_dir = os.path.join(obj_dir, 'mask')

        rgb_paths = glob.glob(os.path.join(rgb_dir, '*.png'))
        asset_obj_dir = os.path.join(asset_dir, '{:02}'.format(obj_id))
        os.system('mkdir -p {}'.format(asset_obj_dir))

        for rgb_path in tqdm.tqdm(rgb_paths):
            mask_path = rgb_path.replace('rgb', 'mask')

            if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                continue

            img = cv2.imread(rgb_path)
            mask = np.array(Image.open(mask_path))
            x, y, w, h = cv2.boundingRect(mask)
            img = img[y:y + h, x:x + w]
            mask = mask[y:y + h, x:x + w]

            img_id = os.path.basename(rgb_path).replace('.png', '')
            rgb_path = os.path.join(asset_obj_dir, img_id + '.png')
            mask_path = os.path.join(asset_obj_dir, img_id + '_mask.png')

            cv2.imwrite(rgb_path, img)
            cv2.imwrite(mask_path, mask)


def asset_to_coco():
    data_root = 'data/tless/train_primesense/assets'
    obj_ids = [i + 1 for i in range(30)]

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    for obj_id in tqdm.tqdm(obj_ids):
        obj_dir = os.path.join(data_root, '{:02}'.format(obj_id))
        mask_paths = glob.glob(os.path.join(obj_dir, '*_mask.png'))
        for mask_path in tqdm.tqdm(mask_paths):
            rgb_path = mask_path.replace('_mask.png', '.png')

            if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                continue

            img_id += 1
            info = {'rgb_path': rgb_path, 'id': img_id}
            images.append(info)

            ann_id += 1
            anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': obj_id, 'id': ann_id}
            annotations.append(anno)

    categories = [{'supercategory': 'none', 'id': obj_id, 'name': str(obj_id)} for obj_id in obj_ids]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}
    anno_path = os.path.join(data_root, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)


def prepare_asset():
    ag_to_asset()
    asset_to_coco()

import tqdm
import os
import glob
import yaml
from PIL import Image
import json
import numpy as np
from lib.datasets.tless import opengl_renderer
from lib.utils import base_utils
import cv2


def test_to_coco():
    data_root = 'data/tless/test_primesense'
    scene_ids = [i + 1 for i in range(20)]

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    for scene_id in tqdm.tqdm(scene_ids):
        scene_dir = os.path.join(data_root, '{:02}'.format(scene_id))
        rgb_dir = os.path.join(scene_dir, 'rgb')
        rgb_paths = glob.glob(os.path.join(rgb_dir, '*.png'))
        gt = yaml.load(open(os.path.join(scene_dir, 'gt.yml')))
        a_pixel_num_dict = base_utils.read_pickle(os.path.join(scene_dir, 'pixel_num.pkl'))
        for rgb_path in tqdm.tqdm(rgb_paths):
            rgb = Image.open(rgb_path)
            img_size = rgb.size
            img_id += 1
            info = {'rgb_path': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
            images.append(info)

            gt_ = gt[int(os.path.basename(rgb_path).replace('.png', ''))]
            mask_path = rgb_path.replace('rgb', 'mask')
            mask = np.array(Image.open(mask_path))
            a_pixel_nums = a_pixel_num_dict[int(os.path.basename(rgb_path).replace('.png', ''))]
            for instance_id, instance_gt in enumerate(gt_):
                obj_id = instance_gt['obj_id']
                mask_id = obj_id * 1000 + instance_id
                mask_ = (mask == mask_id).astype(np.uint8)
                pixel_num = np.sum(mask_)
                a_pixel_num = a_pixel_nums[instance_id]

                if pixel_num / a_pixel_num < 0.1:
                    continue

                ann_id += 1
                bbox = cv2.boundingRect(mask_)
                area = int(np.sum(mask_.astype(np.uint8)))
                anno = {'area': area, 'image_id': img_id, 'bbox': bbox, 'iscrowd': 0, 'category_id': obj_id, 'id': ann_id}
                annotations.append(anno)

    obj_ids = [i + 1 for i in range(30)]
    categories = [{'supercategory': 'none', 'id': obj_id, 'name': str(obj_id)} for obj_id in obj_ids]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}
    anno_path = os.path.join(data_root, 'test.json')
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


def update_mask(depth, w, mask_map, depth_map, obj_id, instance_id):
    col_row = np.argwhere(depth != 0)[:, [1, 0]]
    depth = depth[col_row[:, 1], col_row[:, 0]]
    pixel_depth = depth_map[col_row[:, 1], col_row[:, 0]]
    inds = (depth < pixel_depth)
    col_row = col_row[inds]
    depth = depth[inds]

    mask_map[col_row[:, 1], col_row[:, 0]] = obj_id * 1000 + instance_id
    depth_map[col_row[:, 1], col_row[:, 0]] = depth


def get_mask():
    data_root = 'data/tless/test_primesense'
    scene_ids = [i + 1 for i in range(20)]
    models = get_rendering_model('data/tless/models_cad')

    for scene_id in tqdm.tqdm(scene_ids):
        scene_dir = os.path.join(data_root, '{:02}'.format(scene_id))
        rgb_dir = os.path.join(scene_dir, 'rgb')
        mask_dir = os.path.join(scene_dir, 'mask')
        os.system('mkdir -p {}'.format(mask_dir))

        rgb_paths = glob.glob(os.path.join(rgb_dir, '*.png'))
        gt = yaml.load(open(os.path.join(scene_dir, 'gt.yml')))
        K_info = yaml.load(open(os.path.join(scene_dir, 'info.yml')))
        a_pixel_num_dict = {}

        for rgb_path in tqdm.tqdm(rgb_paths):
            img_id = int(os.path.basename(rgb_path).replace('.png', ''))
            gt_ = gt[img_id]
            w, h = Image.open(rgb_path).size
            K = K_info[img_id]['cam_K']
            K = np.array(K).reshape(3, 3)

            mask_map = np.zeros(shape=[h, w], dtype=np.int16)
            depth_map = 10 * np.ones(shape=[h, w], dtype=np.float32)

            a_pixel_nums = []
            for instance_id, instance_gt in enumerate(gt_):
                R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
                t = np.array(instance_gt['cam_t_m2c']) * 0.001
                pose = np.concatenate([R, t[:, None]], axis=1)

                obj_id = instance_gt['obj_id']
                depth = opengl_renderer.render(models[obj_id], pose, K, w, h)
                update_mask(depth, w, mask_map, depth_map, obj_id, instance_id)
                a_pixel_nums.append(np.sum(depth != 0))

            mask_path = rgb_path.replace('rgb', 'mask')
            Image.fromarray(mask_map).save(mask_path, 'PNG')
            a_pixel_num_dict.update({img_id: a_pixel_nums})

        base_utils.save_pickle(a_pixel_num_dict, os.path.join(scene_dir, 'pixel_num.pkl'))


def record_scene_ann(model_meta, pose_meta, img_id, ann_id, images, annotations):
    corner_3d_dict = model_meta['corner_3d']
    center_3d_dict = model_meta['center_3d']
    fps_3d_dict = model_meta['fps_3d']

    rgb_paths = pose_meta['rgb_paths']
    gt = pose_meta['gt']
    K = pose_meta['K']
    a_pixel_num_dict = pose_meta['a_pixel_num']

    for rgb_path in tqdm.tqdm(rgb_paths):
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        ind = int(os.path.basename(rgb_path).replace('.png', ''))
        K_ = K[ind]['cam_K']
        K_ = np.array(K_).reshape(3, 3)
        gt_ = gt[ind]

        mask_path = rgb_path.replace('rgb', 'mask')
        mask = np.array(Image.open(mask_path))
        a_pixel_nums = a_pixel_num_dict[ind]
        for instance_id, instance_gt in enumerate(gt_):
            obj_id = instance_gt['obj_id']
            mask_id = obj_id * 1000 + instance_id
            mask_ = (mask == mask_id).astype(np.uint8)
            pixel_num = np.sum(mask_)
            a_pixel_num = a_pixel_nums[instance_id]

            if pixel_num / a_pixel_num < 0.1:
                continue

            R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
            t = np.array(instance_gt['cam_t_m2c']) * 0.001
            pose = np.concatenate([R, t[:, None]], axis=1)

            corner_3d = corner_3d_dict[obj_id]
            center_3d = center_3d_dict[obj_id]
            fps_3d = fps_3d_dict[obj_id]

            corner_2d = base_utils.project(corner_3d, K_, pose)
            center_2d = base_utils.project(center_3d[None], K_, pose)[0]
            fps_2d = base_utils.project(fps_3d, K_, pose)

            ann_id += 1
            anno = {}
            anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
            anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
            anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
            anno.update({'K': K_.tolist(), 'pose': pose.tolist()})
            anno.update({'category_id': obj_id, 'image_id': img_id, 'id': ann_id})
            annotations.append(anno)

    return img_id, ann_id


def get_model_corners(model):
    min_x, max_x = np.min(model[:, 0]), np.max(model[:, 0])
    min_y, max_y = np.min(model[:, 1]), np.max(model[:, 1])
    min_z, max_z = np.min(model[:, 2]), np.max(model[:, 2])
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def test_pose_to_coco():
    data_root = 'data/tless/test_primesense'
    scene_ids = [i + 1 for i in range(20)]
    models = get_rendering_model('data/tless/models_cad')
    corner_3d = {i: get_model_corners(v['pts']) / 1000. for i, v in models.items()}
    center_3d = {i: (np.max(v, 0) + np.min(v, 0)) / 2 for i, v in corner_3d.items()}
    fps_3d = {i+1: np.loadtxt('data/tless/farthest/farthest_{:02}.txt'.format(i+1)) for i in range(0, 30)}

    model_meta = {
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    for scene_id in tqdm.tqdm(scene_ids):
        scene_dir = os.path.join(data_root, '{:02}'.format(scene_id))
        rgb_dir = os.path.join(scene_dir, 'rgb')
        rgb_paths = glob.glob(os.path.join(rgb_dir, '*.png'))
        gt = yaml.load(open(os.path.join(scene_dir, 'gt.yml')))
        K_info = yaml.load(open(os.path.join(scene_dir, 'info.yml')))
        a_pixel_num_dict = base_utils.read_pickle(os.path.join(scene_dir, 'pixel_num.pkl'))
        pose_meta = {'rgb_paths': rgb_paths, 'gt': gt, 'K': K_info, 'a_pixel_num': a_pixel_num_dict}

        img_id, ann_id = record_scene_ann(model_meta, pose_meta, img_id, ann_id, images, annotations)

    obj_ids = [i + 1 for i in range(30)]
    categories = [{'supercategory': 'none', 'id': obj_id, 'name': str(obj_id)} for obj_id in obj_ids]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    data_cache_dir = 'data/cache/tless_pose/'
    os.system('mkdir -p {}'.format(data_cache_dir))
    anno_path = os.path.join(data_cache_dir, 'test.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)

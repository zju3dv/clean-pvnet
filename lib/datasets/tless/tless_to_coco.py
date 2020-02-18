import os
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import numpy as np
from PIL import Image
import tqdm
from skimage import measure
import cv2
import json
from lib.utils.base_utils import read_pickle, save_pickle
from lib.utils.linemod.linemod_config import linemod_cls_names, linemod_K, blender_K
import matplotlib.pyplot as plt
import glob
import yaml
from lib.datasets.tless import symmetry_utils


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


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def choose_gt(obj_id, gt, gt_symmetry, ind):
    non_sym = [4, 5, 6, 7, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 25, 26]
    if obj_id in non_sym:
        return gt

    partial_sym = {1: 360, 2: 360, 3: 360, 8: 1080, 9: 215, 24: 215}

    if obj_id not in partial_sym:
        return gt_symmetry

    rng = partial_sym[obj_id]

    if obj_id in [9, 24]:
        if ind < rng:
            return gt_symmetry
        else:
            return gt

    if ind < rng:
        return gt
    else:
        return gt_symmetry


def record_real_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    obj_id = model_meta['obj_id']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']

    obj_dir = os.path.join(data_root, 'train_primesense/{:02}'.format(obj_id))
    rgb_dir = os.path.join(obj_dir, 'rgb')
    mask_dir = os.path.join(obj_dir, 'mask')

    obj_asset_dir = os.path.join(data_root, 'train_primesense/assets', '{:02}'.format(obj_id))

    gt = yaml.load(open(os.path.join(obj_dir, 'gt.yml'), 'r'))
    gt_symmetry = yaml.load(open(os.path.join(obj_dir, 'gt_symmetry.yml'), 'r'))
    K_info = yaml.load(open(os.path.join(obj_dir, 'info.yml'), 'r'))
    inds = list(gt.keys())

    for ind in tqdm.tqdm(inds):
        img_name = '{:04}.png'.format(ind)
        rgb_path = os.path.join(obj_asset_dir, img_name)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        instance_gt = choose_gt(obj_id, gt, gt_symmetry, ind)[ind][0]
        R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
        t = np.array(instance_gt['cam_t_m2c']) * 0.001
        pose = np.concatenate([R, t[:, None]], axis=1)

        K = K_info[ind]['cam_K']
        K = np.array(K).reshape(3, 3)
        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        mask_path = os.path.join(mask_dir, img_name)
        mask = np.array(Image.open(mask_path))
        x, y, w, h = cv2.boundingRect(mask)

        corner_2d = corner_2d - [x, y]
        center_2d = center_2d - [x, y]
        fps_2d = fps_2d - [x, y]

        mask_path = os.path.join(obj_asset_dir, img_name.replace('.png', '_mask.png'))

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_2d': corner_2d.tolist()})
        anno.update({'center_2d': center_2d.tolist()})
        anno.update({'fps_2d': fps_2d.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'real', 'cls': obj_id})
        annotations.append(anno)

    return img_id, ann_id


def choose_render_pose(obj_id, K_P):
    non_sym = [4, 5, 6, 7, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23, 25, 26]
    if obj_id in non_sym or 's_RT' not in K_P:
        return K_P['RT']

    sym = [14, 15, 16, 17, 27, 28, 29, 30]
    if obj_id in sym:
        return K_P['s_RT']

    return K_P['RT']


def record_render_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    obj_id = model_meta['obj_id']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']

    render_dir = os.path.join(data_root, 'renders', str(obj_id))
    ann_num = len(glob.glob(os.path.join(render_dir, '*.pkl')))
    for ind in tqdm.tqdm(range(ann_num)):
        img_name = '{}.png'.format(ind)
        rgb_path = os.path.join(render_dir, img_name)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        K_P = read_pickle(os.path.join(render_dir, '{}_RT.pkl'.format(ind)))
        K = K_P['K']
        pose = choose_render_pose(obj_id, K_P)

        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        mask_path = os.path.join(render_dir, '{}_depth.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': render_dir})
        anno.update({'type': 'render', 'cls': obj_id})
        annotations.append(anno)

    return img_id, ann_id


def _tless_train_to_coco(obj_id):
    data_root = 'data/tless'

    model_path = os.path.join(data_root, 'models_cad', 'obj_{:03}.ply'.format(obj_id))
    renderer = OpenGLRenderer(model_path)

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(os.path.join(data_root, 'farthest', 'farthest_{:02}.txt'.format(obj_id)))

    model_meta = {
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
        'obj_id': obj_id,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    img_id, ann_id = record_real_ann(model_meta, img_id, ann_id, images, annotations)
    img_id, ann_id = record_render_ann(model_meta, img_id, ann_id, images, annotations)

    categories = [{'supercategory': 'none', 'id': 1, 'name': obj_id}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    data_cache_dir = 'data/cache/tless_pose/'
    obj_cache_dir = os.path.join(data_cache_dir, '{:02}'.format(obj_id))
    os.system('mkdir -p {}'.format(obj_cache_dir))
    anno_path = os.path.join(obj_cache_dir, 'train.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)


def tless_train_to_coco():
    _tless_train_to_coco(17)
    # obj_ids = [i + 1 for i in range(30)]
    # for obj_id in tqdm.tqdm(obj_ids):
    #     _tless_train_to_coco(obj_id)


def _handle_real_train_symmetry_pose(obj_id):
    data_root = 'data/tless'

    obj_dir = os.path.join(data_root, 'train_primesense/{:02}'.format(obj_id))
    gt = yaml.load(open(os.path.join(obj_dir, 'gt.yml'), 'r'))
    inds = list(gt.keys())

    for ind in tqdm.tqdm(inds):
        instance_gt = gt[ind][0]
        R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
        R = symmetry_utils.TLESS_rectify(obj_id, R)
        instance_gt['cam_R_m2c'] = R.ravel().tolist()

    yaml.dump(gt, open(os.path.join(obj_dir, 'gt_symmetry.yml'), 'w'))


def _handle_render_train_symmetry_pose(obj_id):
    data_root = 'data/tless'
    render_dir = os.path.join(data_root, 'renders', str(obj_id))

    ann_num = len(glob.glob(os.path.join(render_dir, '*.pkl')))
    for ind in tqdm.tqdm(range(ann_num)):
        pkl_path = os.path.join(render_dir, '{}_RT.pkl'.format(ind))
        K_P = read_pickle(pkl_path)
        pose = K_P['RT']
        symmetry_R = symmetry_utils.TLESS_rectify(obj_id, pose[:, :3])
        K_P['s_RT'] = np.concatenate([symmetry_R, pose[:, 3:]], axis=1)
        save_pickle(K_P, pkl_path)


def _handle_train_symmetry_pose(obj_id):
    _handle_real_train_symmetry_pose(obj_id)
    _handle_render_train_symmetry_pose(obj_id)


def handle_train_symmetry_pose():
    _handle_train_symmetry_pose(17)
    # obj_ids = [i + 1 for i in range(30)]
    # for obj_id in tqdm.tqdm(obj_ids):
    #     _handle_train_symmetry_pose(obj_id)

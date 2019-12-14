import os
from lib.utils.linemod.opengl_renderer import OpenGLRenderer
import numpy as np
from PIL import Image
import tqdm
from skimage import measure
import cv2
import json
from lib.utils.base_utils import read_pickle
from lib.utils.linemod.linemod_config import linemod_cls_names, linemod_K, blender_K
import matplotlib.pyplot as plt
import glob


def read_mask(path, split, cls_idx=1):
    if split == 'train' or split == 'test':
        return (np.asarray(Image.open(path))[:, :, 0] != 0).astype(np.uint8)
    elif split == 'fuse':
        return (np.asarray(Image.open(path)) == cls_idx).astype(np.uint8)
    elif split == 'render':
        return (np.asarray(Image.open(path))).astype(np.uint8)


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


def binary_mask_to_polygon(binary_mask, tolerance=0):
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) <= 4:
            continue
        contour = np.flip(contour, axis=1)
        contour = np.round(np.maximum(contour, 0)).astype(np.int32)
        polygons.append(contour)
    return polygons


def record_occ_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = 'data/occlusion_linemod'
    model_meta['data_root'] = data_root
    cls = model_meta['cls']
    split = model_meta['split']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    inds = np.loadtxt(os.path.join('data/linemod', cls, 'test_occlusion.txt'), np.str)
    inds = [int(os.path.basename(ind).replace('.jpg', '')) for ind in inds]

    rgb_dir = os.path.join(data_root, 'RGB-D/rgb_noseg')
    for ind in tqdm.tqdm(inds):
        img_name = 'color_{:05d}.png'.format(ind)
        rgb_path = os.path.join(rgb_dir, img_name)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose_dir = os.path.join(data_root, 'blender_poses', cls)
        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        pose = np.load(pose_path)

        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        mask_path = os.path.join(data_root, 'masks', cls, '{}.png'.format(ind))
        depth_path = os.path.join(data_root, 'RGB-D', 'depth_noseg',
                'depth_{:05d}.png'.format(ind))

        ann_id += 1
        anno = {'mask_path': mask_path, 'depth_path': depth_path,
                'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'render', 'cls': cls})
        annotations.append(anno)

    return img_id, ann_id


def record_real_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    cls = model_meta['cls']
    split = model_meta['split']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    pose_dir = os.path.join(data_root, cls, 'pose')
    rgb_dir = os.path.join(data_root, cls, 'JPEGImages')

    inds = np.loadtxt(os.path.join(data_root, cls, split + '.txt'), np.str)
    inds = [int(os.path.basename(ind).replace('.jpg', '')) for ind in inds]

    for ind in tqdm.tqdm(inds):
        img_name = '{:06}.jpg'.format(ind)
        rgb_path = os.path.join(rgb_dir, img_name)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose_path = os.path.join(pose_dir, 'pose{}.npy'.format(ind))
        pose = np.load(pose_path)
        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        mask_path = os.path.join(data_root, cls, 'mask', '{:04d}.png'.format(ind))

        ann_id += 1
        depth_path = os.path.join('data/linemod_orig', cls, 'data', 'depth{}.dpt'.format(ind))
        anno = {'mask_path': mask_path, 'depth_path': depth_path,
                'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'real', 'cls': cls})
        annotations.append(anno)

    return img_id, ann_id


def record_fuse_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    cls = model_meta['cls']
    split = model_meta['split']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    fuse_dir = os.path.join(data_root, 'fuse')
    original_K = linemod_K
    ann_num = len(glob.glob(os.path.join(fuse_dir, '*.pkl')))
    cls_idx = linemod_cls_names.index(cls)
    for ind in tqdm.tqdm(range(ann_num)):
        mask_path = os.path.join(fuse_dir, '{}_mask.png'.format(ind))
        mask_real = read_mask(mask_path, 'fuse', cls_idx + 1)
        if (np.sum(mask_real) < 400):
            continue

        img_name = '{}_rgb.jpg'.format(ind)
        rgb_path = os.path.join(fuse_dir, img_name)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}

        begins, poses = read_pickle(os.path.join(fuse_dir, '{}_info.pkl'.format(ind)))
        pose = poses[cls_idx]
        K = original_K.copy()
        K[0, 2] += begins[cls_idx, 1]
        K[1, 2] += begins[cls_idx, 0]

        corner_2d = project(corner_3d, K, pose)
        center_2d = project(center_3d[None], K, pose)[0]
        fps_2d = project(fps_3d, K, pose)

        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': fuse_dir})
        anno.update({'type': 'fuse', 'cls': cls})
        annotations.append(anno)
        images.append(info)

    return img_id, ann_id


def record_render_ann(model_meta, img_id, ann_id, images, annotations):
    data_root = model_meta['data_root']
    cls = model_meta['cls']
    split = model_meta['split']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    render_dir = os.path.join(data_root, 'renders', cls)
    ann_num = len(glob.glob(os.path.join(render_dir, '*.pkl')))
    K = blender_K
    for ind in tqdm.tqdm(range(ann_num)):
        img_name = '{}.jpg'.format(ind)
        rgb_path = os.path.join(render_dir, img_name)
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        pose = read_pickle(os.path.join(render_dir, '{}_RT.pkl'.format(ind)))['RT']

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
        anno.update({'type': 'render', 'cls': cls})
        annotations.append(anno)

    return img_id, ann_id


def _linemod_to_coco(cls, split):
    data_root = 'data/linemod'
    model_path = os.path.join(data_root, cls, cls+'.ply')

    renderer = OpenGLRenderer(model_path)
    K = linemod_K

    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(os.path.join(data_root, cls, 'farthest.txt'))

    model_meta = {
        'K': K,
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
        'cls': cls,
        'split': split
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    if split == 'occ':
        img_id, ann_id = record_occ_ann(model_meta, img_id, ann_id, images, annotations)
    elif split == 'train' or split == 'test':
        img_id, ann_id = record_real_ann(model_meta, img_id, ann_id, images, annotations)

    if split == 'train':
        img_id, ann_id = record_fuse_ann(model_meta, img_id, ann_id, images, annotations)
        img_id, ann_id = record_render_ann(model_meta, img_id, ann_id, images, annotations)

    categories = [{'supercategory': 'none', 'id': 1, 'name': cls}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    anno_path = os.path.join(data_root, cls, split + '.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)


def linemod_to_coco(cfg, only_test=False):
    if not only_test:
        _linemod_to_coco(cfg.cls_type, 'train')
    _linemod_to_coco(cfg.cls_type, 'test')
    _linemod_to_coco(cfg.cls_type, 'occ')

"""
handle_custom_dataset模块
=========================

基于个人数据集生成适用于本项目的信息文件.生成的信息文件包含的内容如下(存放在dict内):

- images(图像信息)-list-dict:
  
  - filename(图像路径):str
  - height(图像高度):int
  - width(图像宽度):int
  - id(图像id):int

- annotations(标注信息)-list-dict:

  - mask_path(掩码路径):str
  - image_id(图像id):int
  - category_id(类别id):int
  - id(标注id):int
  - corner_3d(角点3D坐标):list(8*3)
  - corner_2d(角点2D坐标):list(8*2)
  - center_3d(中心点3D坐标):list(3)
  - center_2d(中心点2D做坐标):list(2)
  - fps_3d(特征点3D坐标):list(8*3)
  - fps_2d(特征点2D坐标):list(8*2)
  - K(相机内参矩阵):list(3*3)
  - pose(相机坐标系到模型坐标系/世界坐标系的位姿矩阵):list(3*4)
  - data_root(图像根目录):str
  - type(数据集类型):str(real)
  - cls(类别):str(cat)

- categories(类别信息)-list-dict:

  - supercategory(父类别):none
  - id(类别id):int(1)
  - name(类别名):str(cat)

.. note:: 待处理的个人数据集应包括以下内容:

          1. model.ply:目标物体的三维模型
          2. camera.txt:相机内参矩阵(可通过numpy.loadtxt读取)
          3. diameter.txt:目标物体的直径
          4. rgb:包含rgb图像的文件夹
          5. pose:包含图像位姿信息的文件夹(可通过numpy.load读取)
          6. mask:包含图像掩码信息的文件夹
"""
# 标准库
import os
import json
from pathlib import Path
# 第三方库
import tqdm
import numpy as np
from PIL import Image
from plyfile import PlyData
# 自建库
from lib.csrc.fps import fps_utils
from lib.utils import base_utils
from lib.utils.linemod.opengl_renderer import OpenGLRenderer

def read_ply_points(ply_path):
    """
    read_ply_points 读取指定ply文件中保存的3D点的集合(3D点云)

    :param ply_path: ply文件路径
    :type ply_path: str
    :return: 物体的3D点云
    :rtype: narray
    """
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data     # 读取ply文件的point elements
    points = np.stack([data['x'], data['y'], data['z']], axis=1)    # 读取3D点的x, y, z坐标
    return points


def sample_fps_points(data_root):
    """
    sample_fps_points 读取data_root目录下的model.ply文件,并基于fps采集其中相对距离最远的8个点(fps_points),
    最后将fps_points保存至fps.txt文件内

    :param data_root: 数据集目录(ply文件所在的根目录)
    :type data_root: str
    """
    ply_path = os.path.join(data_root, 'model.ply')
    ply_points = read_ply_points(ply_path)
    fps_points = fps_utils.farthest_point_sampling(ply_points, 8, True) # 基于最远点采样(fps),选取点云中中相对距离最远的8个点
    np.savetxt(os.path.join(data_root, 'fps.txt'), fps_points)  # 将fps_points保存在fps.txt文件中


def get_model_corners(model):
    """
    get_model_corners 获得外切于模型的矩形框的8个顶点坐标

    :param model: 模型表面的vertices坐标
    :type model: narray(n*3)
    :return: 边界框的8个顶点坐标
    :rtype: narray(8*3)
    """
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


def record_ann(path_list, model_meta, img_id, ann_id, images, annotations):
    """
    record_ann 记录数据集的image和annotation信息

    :param model_meta: 模型元数据:数据集根目录,角点3D坐标,中心点3D坐标,特征点3D坐标,相机内参矩阵
    :type model_meta: dict
    :param img_id: 图像id
    :type img_id: int
    :param ann_id: 标注id
    :type ann_id: int
    :param images: 数据集的图像信息
    :type images: list
    :param annotations: 数据集的标注信息
    :type annotations: list
    :return: 数据集的img个数和ann个数
    :rtype: tuple(int, int)
    """
    data_root = model_meta['data_root']
    corner_3d = model_meta['corner_3d']
    center_3d = model_meta['center_3d']
    fps_3d = model_meta['fps_3d']
    K = model_meta['K']

    pose_dir = os.path.join(data_root, 'pose')
    rgb_dir = os.path.join(data_root, 'rgb')
    mask_dir = os.path.join(data_root, 'mask')

    for path in tqdm.tqdm(path_list):
        rgb_path = os.path.join(rgb_dir, '/'.join(path.split('/')[-2:]))

        # 记录当前图片的image信息
        rgb = Image.open(rgb_path)
        img_size = rgb.size
        img_id += 1
        info = {'file_name': rgb_path, 'height': img_size[1], 'width': img_size[0], 'id': img_id}
        images.append(info)

        # 计算3D坐标的2D投影
        pose_path = os.path.join(pose_dir, ('/'.join(path.split('/')[-2:]))[:-3]+'npy')
        pose = np.load(pose_path)  # 从相机坐标系到模型当前位置的位姿变换矩阵
        corner_2d = base_utils.project(corner_3d, K, pose)  # 计算当前图像中的角点坐标
        center_2d = base_utils.project(center_3d[None], K, pose)[0]  # 计算当前图像中的中心点坐标
        fps_2d = base_utils.project(fps_3d, K, pose)  # 计算当前图像中的特征点坐标

        mask_path = os.path.join(mask_dir, ('/'.join(path.split('/')[-2:]))[:-3]+'png')

        # 记录当前图片的annotation信息
        ann_id += 1
        anno = {'mask_path': mask_path, 'image_id': img_id, 'category_id': 1, 'id': ann_id}
        anno.update({'corner_3d': corner_3d.tolist(), 'corner_2d': corner_2d.tolist()})
        anno.update({'center_3d': center_3d.tolist(), 'center_2d': center_2d.tolist()})
        anno.update({'fps_3d': fps_3d.tolist(), 'fps_2d': fps_2d.tolist()})
        anno.update({'K': K.tolist(), 'pose': pose.tolist()})
        anno.update({'data_root': rgb_dir})
        anno.update({'type': 'render', 'cls': 'charger'})
        annotations.append(anno)

    return img_id, ann_id


def custom_to_coco(data_root='data/custom',path_list=[],kind='train'):
    """
    custom_to_coco 处理数据集,并生成PVNet需要的信息文件

    :param data_root: 数据集目录
    :type data_root: str
    """
    model_path = os.path.join(data_root, 'model.ply')

    # 加载模型的ply文件和相机内参矩阵K
    renderer = OpenGLRenderer(model_path)
    K = np.loadtxt(os.path.join(data_root, 'camera.txt'))

    # 获取模型的角点,中心点,特征点
    model = renderer.model['pts'] / 1000
    corner_3d = get_model_corners(model)
    center_3d = (np.max(corner_3d, 0) + np.min(corner_3d, 0)) / 2
    fps_3d = np.loadtxt(os.path.join(data_root, 'fps.txt'))

    """模型元数据"""
    model_meta = {
        'K': K,
        'corner_3d': corner_3d,
        'center_3d': center_3d,
        'fps_3d': fps_3d,
        'data_root': data_root,
    }

    img_id = 0
    ann_id = 0
    images = []
    annotations = []
    
    # 生成PVNet需要的相关信息:images(图像信息),annotations(标注信息),categories(类别信息)
    img_id, ann_id = record_ann(path_list, model_meta, img_id, ann_id, images, annotations)
    categories = [{'supercategory': 'none', 'id': 1, 'name': 'charger'}]
    instance = {'images': images, 'annotations': annotations, 'categories': categories}

    # 将数据集的信息写入到数据集目录下的train.json文件内
    anno_path = os.path.join(data_root, kind+'.json')
    with open(anno_path, 'w') as f:
        json.dump(instance, f)

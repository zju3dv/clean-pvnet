import pycocotools.mask as mask_utils
import numpy as np
from plyfile import PlyData
from PIL import Image


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


def coco_poly_to_mask(poly, h, w):
    rles = mask_utils.frPyObjects(poly, h, w)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask


def compute_vertex(mask, kpt_2d):
    """
    compute_vertex 计算图像中各个像素指向所有特征点 ``kpt_2d`` 的单位向量

    :param mask: 图像掩码
    :type mask: narray
    :param kpt_2d: 2D关键点
    :type kpt_2d: narray
    :return: 图像像素指向特征点的单位向量
    :rtype: narray(dtype=np.float32)

    .. note:: 计算的单位向量时分为三种情况:

              1. 当前像素属于目标实例：正常计算单位向量
              2. 当前像素即为特征点：单位向量为(0, 0)
              3. 当前像素不属于目标实例：单位向量为(0, 0)
    """    
    # 取图像大小(h, w)，D关键点个数m，图像中实例的所有像素坐标xy(注意是xy坐标，即宽*高)
    h, w = mask.shape
    m = kpt_2d.shape[0]
    xy = np.argwhere(mask == 1)[:, [1, 0]]  # numpy.argwhere返回数组非零元素的下标(以元素分组)，与numpy.nonzero区分

    # 分别计算每个特征点与实例上所有像素的欧式距离
    vertex = kpt_2d[None] - xy[:, None]                     # 特征点与实例像素2D坐标之差(Δx, Δy)
    norm = np.linalg.norm(vertex, axis=2, keepdims=True)    # 计算vertex/每个(Δx, Δy)的2范数(即欧式距离)
    norm[norm < 1e-3] += 1e-3                               # norm元素最小值为0.001，避免下一步产生除零错误
    vertex = vertex / norm                                  # 计算单位向量(Δx/L, Δy/L), L = sqrt(Δx²+Δy²)

    # 将计算的结果vertex保存在(h, w, m*2)的数组内
    vertex_out = np.zeros([h, w, m, 2], np.float32)
    vertex_out[xy[:, 1], xy[:, 0]] = vertex
    vertex_out = np.reshape(vertex_out, [h, w, m * 2])

    return vertex_out


def get_ply_model(model_path):
    """
    get_ply_model 读取指定的三维ply模型,并返回模型上的三维点坐标

    :param model_path: 三维模型的路径
    :type model_path: str
    :return: 模型上的三维坐标(n*3)
    :rtype: narray
    """
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    model = np.stack([x, y, z], axis=-1)
    return model


def read_linemod_mask(path, ann_type, cls_idx):
    """
    read_linemod_mask 读取linemod数据集图像的掩码

    :param path: mask路径
    :type path: str
    :param ann_type: 图像类型(real, fuse, render)
    :type ann_type: str
    :param cls_idx: 类别id
    :type cls_idx: int
    :return: 图像掩码
    :rtype: array(type=np.uint8)
    """    
    if ann_type == 'real':
        mask = np.array(Image.open(path))
        if len(mask.shape) == 3:    # 若图像是真实彩色(real)图像，则仅取第一个通道的掩码即可
            return (mask[..., 0] != 0).astype(np.uint8)
        else:
            return (mask != 0).astype(np.uint8)
    elif ann_type == 'fuse':        # 若图像是混合(fuse)图像，仅返回当前类别的掩码
        return (np.asarray(Image.open(path)) == cls_idx).astype(np.uint8)
    elif ann_type == 'render':      # 若图像是渲染(render)图像，转换为narray直接返回
        return (np.asarray(Image.open(path))).astype(np.uint8)


def read_tless_mask(ann_type, path):
    if ann_type == 'real':
        return (np.asarray(Image.open(path))).astype(np.uint8)
    elif ann_type == 'render':
        depth = np.asarray(Image.open(path))
        return (depth != 65535).astype(np.uint8)

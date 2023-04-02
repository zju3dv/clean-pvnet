import pickle
import os
import numpy as np


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def project(xyz, K, RT):
    """
    project 将世界坐标系中的坐标xyz转换到图像坐标系中,并返回图像坐标xy

    :param xyz: 世界坐标[N, 3]
    :type xyz: narray
    :param K: 相机内参矩阵[3, 3]
    :type K: narry
    :param RT: 相机坐标系到世界坐标系的位姿变换矩阵[3, 4]
    :type RT: narray
    :return: 世界坐标在图像坐标系中的位置
    :rtype: narray
    """    
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T  # 将世界坐标xyz转换至相机坐标系下
    xyz = np.dot(xyz, K.T)  # 再结合相机内参,得到xyz在图像坐标系中的齐次坐标
    xy = xyz[:, :2] / xyz[:, 2:]  # 最后再将齐次坐标转换为普通坐标的形式                  
    return xy

import OpenEXR
import Imath
import png
import os
import pickle
import numpy as np
from PIL import Image
from transforms3d.euler import mat2euler, euler2mat
from plyfile import PlyData


def save_string(string, path):
    with open(path, 'w') as f:
        f.write(string)


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def exr_to_png(exr_path):
    depth_path = exr_path.replace('.png0001.exr', '.png')
    pose_path = exr_path.replace('_depth.png0001.exr', '_RT.pkl')
    exr_image = OpenEXR.InputFile(exr_path)
    dw = exr_image.header()['dataWindow']
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    def read_exr(s, width, height):
        mat = np.fromstring(s, dtype=np.float32)
        mat = mat.reshape(height, width)
        return mat

    def saveUint16(z, path):
        # Use pypng to write zgray as a grayscale PNG.
        with open(path, 'wb') as f:
            writer = png.Writer(
                width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
            zgray2list = z.tolist()
            writer.write(f, zgray2list)

    dmap, _, _ = [read_exr(s, width, height)
                  for s in exr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
    dmap *= 10
    row_col = np.argwhere(dmap != 10)
    xy = row_col[:, [1, 0]].astype(np.float32)  # flip xy
    D = dmap[dmap != 10]
    K_P = read_pickle(pose_path)
    K = K_P['K']
    L = np.sqrt(np.power(xy[:, 0] - K[0, 2] + 0.5, 2) + np.power(
        xy[:, 1] - K[1, 2] + 0.5, 2) + np.power(K[0, 0], 2))
    d = D / L * K[0, 0]
    dmap[row_col[:, 0], row_col[:, 1]] = d

    saveUint16((dmap / 10 * 65535).astype(np.uint16), depth_path)
    exr_image.close()
    os.system('rm {}'.format(exr_path))


class ViewpointSampler(object):

    @staticmethod
    def haversine(viewpoints):
        """
        viewpoints: [N, 2] (in decimal degrees)
        Calculate the great circle distances of viewpoints from [0, 0]
        reference: https://en.wikipedia.org/wiki/Great-circle_distance
        """
        # convert decimal degrees to radians
        viewpoints = np.deg2rad(viewpoints)
        # haversine formula
        lat1 = 0
        lat2 = viewpoints[:, 1]
        dlon = viewpoints[:, 0]
        dlat = viewpoints[:, 1]
        a = np.sin(dlat / 2)**2 + np.cos(
            lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return c

    @staticmethod
    def sample_sphere(num_samples):
        """ sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        begin_elevation = 0
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []

        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(np.rad2deg(2 * np.pi * n * phi % (2 * np.pi)))
            elevations.append(np.rad2deg(np.arcsin(z)))

        tilts = np.zeros_like(azimuths)
        viewpoints = np.stack([azimuths, elevations, tilts], axis=-1)
        inds = np.lexsort(
            [viewpoints[:, 2], viewpoints[:, 1], viewpoints[:, 0]])
        viewpoints = viewpoints[inds]

        return viewpoints


class Projector(object):
    intrinsic_matrix = {
        'blender': np.array([[700.,    0.,  320.],
                             [0.,  700.,  240.],
                             [0.,    0.,    1.]])
    }

    @staticmethod
    def project(pts_3d, RT, K_type):
        pts_2d = np.matmul(pts_3d, RT[:, :3].T) + RT[:, 3:].T
        pts_2d = np.matmul(pts_2d, Projector.intrinsic_matrix[K_type].T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d

    @staticmethod
    def project_K(pts_3d, RT, K):
        pts_2d = np.matmul(pts_3d, RT[:, :3].T) + RT[:, 3:].T
        pts_2d = np.matmul(pts_2d, K.T)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
        return pts_2d


def recover_3d_point_cloud(dmap_path, krt_path):
    KRT = read_pickle(krt_path)
    K = KRT['K']

    dmap = np.array(Image.open(dmap_path)) / 65535 * 10
    row_col = np.argwhere(dmap != 10)
    xy = row_col[:, [1, 0]].astype(np.float32)
    d = dmap[dmap != 10][..., np.newaxis]
    xy *= d
    xyz = np.concatenate([xy, d], axis=1)
    xyz = np.dot(xyz, np.linalg.inv(K).T)

    rt = KRT['RT']
    xyz = np.dot(xyz - rt[:, 3], np.linalg.inv(rt[:, :3]).T)

    return xyz


def blender_pose_to_blender_euler(pose):
    euler = [r / np.pi * 180 for r in mat2euler(pose, axes='szxz')]
    euler[0] = -(euler[0] + 90) % 360
    euler[1] = euler[1] - 90
    return np.array(euler)


def blender_euler_to_blender_pose(euler):
    azi = -euler[0] - 90
    ele = euler[1] + 90
    theta = euler[2]
    pose = euler2mat(azi, ele, theta, axes='szxz')
    return pose


def validate_pose(img, pts_3d, pose, K):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    pts_2d = Projector.project_K(pts_3d, pose, K)
    plt.plot(pts_2d[:, 0], pts_2d[:, 1], '.')
    plt.show()


def read_ply(ply_path):
    ply = PlyData.read(ply_path)
    data = ply.elements[0].data
    x = data['x']
    y = data['y']
    z = data['z']
    return np.stack([x, y, z], axis=-1)


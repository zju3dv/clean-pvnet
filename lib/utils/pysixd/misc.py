# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import os
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import distance

def ensure_dir(path):
    """
    Ensures that the specified directory exists.

    :param path: Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def draw_rect(vis, rect, color=(255, 255, 255)):
    vis_pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(vis_pil)
    draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
                   outline=color, fill=None)
    del draw
    return np.asarray(vis_pil)

def project_pts(pts, K, R, t):
    assert(pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T

def norm_depth(depth, valid_start=0.2, valid_end=1.0):
    mask = depth > 0
    depth_n = depth.astype(np.float)
    depth_n[mask] -= depth_n[mask].min()
    depth_n[mask] /= depth_n[mask].max() / (valid_end - valid_start)
    depth_n[mask] += valid_start
    return depth_n

def depth_im_to_dist_im(depth_im, K):
    """
    Converts depth image to distance image.

    :param depth_im: Input depth image, where depth_im[y, x] is the Z coordinate
    of the 3D point [X, Y, Z] that projects to pixel [x, y], or 0 if there is
    no such 3D point (this is a typical output of the Kinect-like sensors).
    :param K: Camera matrix.
    :return: Distance image dist_im, where dist_im[y, x] is the distance from
    the camera center to the 3D point [X, Y, Z] that projects to pixel [x, y],
    or 0 if there is no such 3D point.
    """
    xs = np.tile(np.arange(depth_im.shape[1]), [depth_im.shape[0], 1])
    ys = np.tile(np.arange(depth_im.shape[0]), [depth_im.shape[1], 1]).T

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

    dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)
    return dist_im

def rgbd_to_point_cloud(K, depth, rgb=np.array([])):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    pts_im = np.vstack([us, vs]).T
    if rgb != np.array([]):
        colors = rgb[vs, us, :]
    else:
        colors = None
    return pts, colors, pts_im

def clip_pt_to_im(pt, im_size):
    pt_c = [min(max(pt[0], 0), im_size[0] - 1),
            min(max(pt[1], 0), im_size[1] - 1)]
    return pt_c

def calc_2d_bbox(xs, ys, im_size=None, clip=False):
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    if clip:
        assert(im_size is not None)
        bb_tl = clip_pt_to_im(bb_tl, im_size)
        bb_br = clip_pt_to_im(bb_br, im_size)
    return [bb_tl[0], bb_tl[1], bb_br[0] - bb_tl[0], bb_br[1] - bb_tl[1]]

def calc_pose_2d_bbox(model, im_size, K, R_m2c, t_m2c):
    pts_im = project_pts(model['pts'], K, R_m2c, t_m2c)
    pts_im = np.round(pts_im).astype(np.int)
    return calc_2d_bbox(pts_im[:, 0], pts_im[:, 1], im_size)

def crop_im(im, roi):
    if im.ndim == 3:
        crop = im[max(roi[1], 0):min(roi[1] + roi[3] + 1, im.shape[0]),
               max(roi[0], 0):min(roi[0] + roi[2] + 1, im.shape[1]), :]
    else:
        crop = im[max(roi[1], 0):min(roi[1] + roi[3] + 1, im.shape[0]),
               max(roi[0], 0):min(roi[0] + roi[2] + 1, im.shape[1])]
    return crop

def paste_im(src, trg, pos):
    """
    Pastes src to trg with the top left corner at pos.
    """
    assert(src.ndim == trg.ndim)

    # Size of the region to be pasted
    w = min(src.shape[1], trg.shape[1] - pos[0])
    h = min(src.shape[0], trg.shape[0] - pos[1])

    if src.ndim == 3:
        trg[pos[1]:(pos[1] + h), pos[0]:(pos[0] + w), :] = src[:h, :w, :]
    else:
        trg[pos[1]:(pos[1] + h), pos[0]:(pos[0] + w)] = src[:h, :w]

def paste_im_mask(src, trg, pos, mask):
    assert(src.ndim == trg.ndim)
    assert(src.shape[:2] == mask.shape[:2])
    src_pil = Image.fromarray(src)
    trg_pil = Image.fromarray(trg)
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    trg_pil.paste(src_pil, pos, mask_pil)
    trg[:] = np.array(trg_pil)[:]

def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T

def calc_pts_diameter(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set).

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    diameter = -1
    for pt_id in range(pts.shape[0]):
        #if pt_id % 1000 == 0: print(pt_id)
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter

def calc_pts_diameter2(pts):
    """
    Calculates diameter of a set of points (i.e. the maximum distance between
    any two points in the set). Faster but requires more memory than
    calc_pts_diameter.

    :param pts: nx3 ndarray with 3D points.
    :return: Diameter.
    """
    dists = distance.cdist(pts, pts, 'euclidean')
    diameter = np.max(dists)
    return diameter

from lib.csrc.uncertainty_pnp._ext import lib, ffi
import numpy as np
import cv2


def uncertainty_pnp(points_2d, weights_2d, points_3d, camera_matrix):
    '''
    :param points_2d:           [pn,2]
    :param weights_2d:          [pn,3] wxx,wxy,wyy
    :param points_3d:           [pn,3]
    :param camera_matrix:       [3,3]
    :return:
    '''
    pn = points_2d.shape[0]
    assert(points_3d.shape[0] == pn and pn >= 4)

    try:
        dist_coeffs = uncertainty_pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype=np.float64)

    points_3d = points_3d.astype(np.float64)
    points_2d = points_2d.astype(np.float64)
    weights_2d = weights_2d.astype(np.float64)
    camera_matrix = camera_matrix.astype(np.float64)

    idxs = np.argsort(weights_2d[:, 0]+weights_2d[:, 1])[-4:]

    _, R_exp, t = cv2.solvePnP(np.expand_dims(points_3d[idxs,:], 0),
                             np.expand_dims(points_2d[idxs,:], 0),
                             camera_matrix, dist_coeffs, None, None, False, flags=cv2.SOLVEPNP_P3P)

    if pn == 4:
        # no other points
        R, _ = cv2.Rodrigues(R_exp)
        Rt = np.concatenate([R, t], axis=-1)
        return Rt

    points_2d = np.ascontiguousarray(points_2d, np.float64)
    points_3d = np.ascontiguousarray(points_3d, np.float64)
    weights_2d = np.ascontiguousarray(weights_2d, np.float64)
    camera_matrix = np.ascontiguousarray(camera_matrix, np.float64)
    init_rt = np.ascontiguousarray(np.concatenate([R_exp, t], 0), np.float64)

    points_2d_ptr = ffi.cast('double*', points_2d.ctypes.data)
    points_3d_ptr = ffi.cast('double*', points_3d.ctypes.data)
    weights_3d_ptr = ffi.cast('double*', weights_2d.ctypes.data)
    camera_matrix_ptr = ffi.cast('double*', camera_matrix.ctypes.data)
    init_rt_ptr = ffi.cast('double*', init_rt.ctypes.data)
    result_rt = np.empty([6], np.float64)
    result_rt_ptr = ffi.cast('double*', result_rt.ctypes.data)

    lib.uncertainty_pnp(points_2d_ptr, points_3d_ptr, weights_3d_ptr, camera_matrix_ptr, init_rt_ptr, result_rt_ptr, pn)

    R, _ = cv2.Rodrigues(result_rt[:3])
    Rt = np.concatenate([R, result_rt[3:, None]], axis=-1)
    return Rt


def uncertainty_pnp_v2(points_2d, covars, points_3d, camera_matrix, type='single'):
    '''
    :param points_2d:           [pn,2]
    :param covars:              [pn,2,2]
    :param points_3d:           [pn,3]
    :param camera_matrix:       [3,3]
    :return:
    '''
    pn = points_2d.shape[0]
    assert(points_3d.shape[0] == pn and pn >= 4 and covars.shape[0] == pn)

    points_3d = points_3d.astype(np.float64)
    points_2d = points_2d.astype(np.float64)
    camera_matrix = camera_matrix.astype(np.float64)

    weights_2d = []
    for pi in range(pn):
        weight = 0.0
        if covars[pi, 0, 0] < 1e-5:
            weights_2d.append(weight)
        else:
            weight = np.max(np.linalg.eigvals(covars[pi]))
            weights_2d.append(1.0/weight)
    weights_2d = np.asarray(weights_2d, np.float64)

    try:
        dist_coeffs = uncertainty_pnp.dist_coeffs
    except:
        dist_coeffs = np.zeros(shape=[8, 1], dtype=np.float64)

    idxs = np.argsort(weights_2d)[-4:]
    _, R_exp, t = cv2.solvePnP(np.expand_dims(points_3d[idxs,:], 0),
                             np.expand_dims(points_2d[idxs,:], 0),
                             camera_matrix, dist_coeffs, None, None, False, flags=cv2.SOLVEPNP_P3P)

    if pn == 4:
        # no other points
        R, _ = cv2.Rodrigues(R_exp)
        Rt = np.concatenate([R, t], axis=-1)
        return Rt

    points_2d = np.ascontiguousarray(points_2d, np.float64)
    points_3d = np.ascontiguousarray(points_3d, np.float64)
    weights_2d = weights_2d[:, None]
    weights_2d = np.concatenate([weights_2d, np.zeros([pn, 1]), weights_2d], 1)
    weights_2d = np.ascontiguousarray(weights_2d, np.float64)
    camera_matrix = np.ascontiguousarray(camera_matrix, np.float64)
    init_rt = np.ascontiguousarray(np.concatenate([R_exp, t], 0), np.float64)

    points_2d_ptr = ffi.cast('double*', points_2d.ctypes.data)
    points_3d_ptr = ffi.cast('double*', points_3d.ctypes.data)
    weights_3d_ptr = ffi.cast('double*', weights_2d.ctypes.data)
    camera_matrix_ptr = ffi.cast('double*', camera_matrix.ctypes.data)
    init_rt_ptr = ffi.cast('double*', init_rt.ctypes.data)
    result_rt = np.empty([6], np.float64)
    result_rt_ptr = ffi.cast('double*', result_rt.ctypes.data)

    lib.uncertainty_pnp(points_2d_ptr, points_3d_ptr, weights_3d_ptr, camera_matrix_ptr, init_rt_ptr, result_rt_ptr, pn)

    R, _ = cv2.Rodrigues(result_rt[:3])
    Rt = np.concatenate([R, result_rt[3:, None]], axis=-1)
    return Rt

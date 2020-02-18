from lib.csrc.nn._ext import lib, ffi
import numpy as np


def find_nearest_point_idx(ref_pts, que_pts):
    assert(ref_pts.shape[1] == que_pts.shape[1] and 1 < que_pts.shape[1] <= 3)
    pn1 = ref_pts.shape[0]
    pn2 = que_pts.shape[0]
    dim = ref_pts.shape[1]

    ref_pts = np.ascontiguousarray(ref_pts[None,:,:], np.float32)
    que_pts = np.ascontiguousarray(que_pts[None,:,:], np.float32)
    idxs = np.zeros([1, pn2], np.int32)

    ref_pts_ptr = ffi.cast('float *', ref_pts.ctypes.data)
    que_pts_ptr = ffi.cast('float *', que_pts.ctypes.data)
    idxs_ptr = ffi.cast('int *', idxs.ctypes.data)
    lib.findNearestPointIdxLauncher(ref_pts_ptr, que_pts_ptr, idxs_ptr, 1, pn1, pn2, dim, 0)

    return idxs[0]

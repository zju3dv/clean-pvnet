# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np

def estimate_visib_mask(d_test, d_model, delta):
    """
    Estimation of visibility mask.

    :param d_test: Distance image of the test scene.
    :param d_model: Rendered distance image of the object model.
    :param delta: Tolerance used in the visibility test.
    :return: Visibility mask.
    """
    assert(d_test.shape == d_model.shape)
    mask_valid = np.logical_and(d_test > 0, d_model > 0)

    d_diff = d_model.astype(np.float32) - d_test.astype(np.float32)
    visib_mask = np.logical_and(d_diff <= delta, mask_valid)

    return visib_mask

def estimate_visib_mask_gt(d_test, d_gt, delta):
    visib_gt = estimate_visib_mask(d_test, d_gt, delta)
    return visib_gt

def estimate_visib_mask_est(d_test, d_est, visib_gt, delta):
    visib_est = estimate_visib_mask(d_test, d_est, delta)
    visib_est = np.logical_or(visib_est, np.logical_and(visib_gt, d_est > 0))
    return visib_est

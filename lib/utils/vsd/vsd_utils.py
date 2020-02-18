import numpy as np
from . import renderer, misc, visibility


def vsd(dist_est, dist_gt, dist_test, visib_gt, delta, tau, cost_type='tlinear'):
    """
    Visible Surface Discrepancy.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param tau: Misalignment tolerance.
    :param cost_type: Pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
        'step' - Used for SIXD Challenge 2017. It is easier to interpret.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    # Visibility mask of the model in the estimated pose
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    if cost_type == 'step':
        costs = costs >= tau
    elif cost_type == 'tlinear': # Truncated linear
        costs *= (1.0 / tau)
        costs[costs > 1.0] = 1.0
    else:
        print('Error: Unknown pixel matching cost.')
        exit(-1)

    # costs_vis = np.ones(dist_gt.shape)
    # costs_vis[visib_inter] = costs
    # import matplotlib.pyplot as plt
    # plt.matshow(costs_vis)
    # plt.colorbar()
    # plt.show()

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
    else:
        e = 1.0
    return e


def _vsd(R_est, t_est, R_gt, t_gt, model, depth_test, K, delta, tau,
        cost_type='tlinear'):
    """
    Visible Surface Discrepancy.
    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param depth_test: Depth image of the test scene.
    :param K: Camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param tau: Misalignment tolerance.
    :param cost_type: Pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
        'step' - Used for SIXD Challenge 2017. It is easier to interpret.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    im_size = (depth_test.shape[1], depth_test.shape[0])

    # Render depth images of the model in the estimated and the ground truth pose
    depth_est = renderer.render(model, im_size, K, R_est, t_est, clip_near=100,
                                clip_far=10000, mode='depth')

    depth_gt = renderer.render(model, im_size, K, R_gt, t_gt, clip_near=100,
                               clip_far=10000, mode='depth')

    # Convert depth images to distance images
    dist_test = misc.depth_im_to_dist_im(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im(depth_est, K)

    # Visibility mask of the model in the ground truth pose
    visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta)

    # Visibility mask of the model in the estimated pose
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    if cost_type == 'step':
        costs = costs >= tau
    elif cost_type == 'tlinear': # Truncated linear
        costs *= (1.0 / tau)
        costs[costs > 1.0] = 1.0
    else:
        print('Error: Unknown pixel matching cost.')
        exit(-1)

    # costs_vis = np.ones(dist_gt.shape)
    # costs_vis[visib_inter] = costs
    # import matplotlib.pyplot as plt
    # plt.matshow(costs_vis)
    # plt.colorbar()
    # plt.show()

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
    else:
        e = 1.0
    return e

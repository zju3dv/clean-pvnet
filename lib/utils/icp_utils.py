import numpy as np
import time
import functools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.neighbors import NearestNeighbors
from lib.utils.meshrenderer import meshrenderer
import open3d as o3d
from lib.config import cfg
from lib.utils.pysixd import transform, misc

# Constants

N = 3000  # number of random points in the dataset
dim = 3  # number of dimensions of the points
verbose = False
# max_mean_dist_factor = 2.0
angle_change_limit = 20 * np.pi / 180.  # = 20 deg #0.5236=30 deg

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def best_fit_transform(A, B, depth_only=False, no_depth=False):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    if depth_only == True and no_depth == False:
        R = np.eye(3)
        t = centroid_B.T - centroid_A.T
        #t = np.array([0, 0, t[2]])
    else:
        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = centroid_B.T - np.dot(R, centroid_A.T)
        if no_depth == True and depth_only == False:
            t = np.array([t[0], t[1], 0])

    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=200, tolerance=0.001, verbose=False, depth_only=False, no_depth=False):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape



    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    if verbose:
        plt.clf()
        fig = plt.figure(1)
        ax = Axes3D(fig)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]] * 3)
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], label='initial', marker='.', c='green')
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], label='target', marker='.', c='blue')

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T, depth_only=depth_only, no_depth=no_depth)

        # update the current source
        src = np.dot(T, src)

        mean_error = np.mean(distances)
        # print mean_error
        # check error
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T, depth_only=depth_only, no_depth=no_depth)

    if verbose:
        anim = ax.scatter(src[0, :], src[1, :], src[2, :], label='estimated', marker='.', c='red')
        plt.legend()
        plt.show()

    return T, distances, i


class SynRenderer(object):
    def __init__(self, class_type):
        MODEL_PATH = os.path.join('data/linemod', 'cad/cad.ply')
        self.model_path = MODEL_PATH.replace('cad', class_type)
        self.renderer

    @lazy_property
    def renderer(self):
        #import ipdb; ipdb.set_trace()
        # if self.model == 'cad':
        # if self.model == 'reconst':
        return meshrenderer.Renderer([self.model_path],1,'.',1000)

    def generate_synthetic_depth(self, K_test, R_est, t_est, test_shape, display=False):
        # renderer = meshrenderer.Renderer(['/data/SLC_precise_blue.ply'],1,'.',1)
        # R = transform.random_rotation_matrix()[:3,:3]
        H_test, W_test = test_shape[:2]
        grey, depth_x = self.renderer.render(
            obj_id=0,
            W=W_test,
            H=H_test,
            K=K_test,
            R=R_est,
            t=t_est,
            near=10,
            far=10000,
            random_light=False
        )
        if display:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.imshow(depth_x)
            plt.subplot(122)
            plt.imshow(grey)
            plt.show()
        pts = misc.rgbd_to_point_cloud(K_test, depth_x)[0]

        return pts

    def render_trafo(self, K_test, R_est, t_est, test_shape, downSample=1):
        W_test, H_test = test_shape[:2]

        bgr, depth_x = self.renderer.render(
            obj_id=0,
            W=W_test,  # /downSample,
            H=H_test,  # /downSample,
            K=K_test,
            R=R_est,
            t=np.array(t_est),
            near=10,
            far=10000,
            random_light=False
        )
        return bgr

def icp_refinement(depth_crop, icp_renderer, R_est, t_est, K_test, test_render_dims, depth_only=False, no_depth=False,
                   max_mean_dist_factor=2.0):
    synthetic_pts = icp_renderer.generate_synthetic_depth(K_test, R_est, t_est, depth_crop.shape)
    centroid_synthetic_pts = np.mean(synthetic_pts, axis=0)
    try:
        max_mean_dist = np.max(np.linalg.norm(synthetic_pts - centroid_synthetic_pts, axis=1))
    except:
        return (R_est, t_est)

    real_depth_pts = misc.rgbd_to_point_cloud(K_test, depth_crop)[0]
    real_synmean_dist = np.linalg.norm(real_depth_pts - centroid_synthetic_pts, axis=1)
    real_depth_pts = real_depth_pts[real_synmean_dist < max_mean_dist_factor * max_mean_dist]
    if len(real_depth_pts) < len(synthetic_pts) / 20.:
        print('not enough visible points')
        R_refined = R_est
        t_refined = t_est
    else:
        sub_idcs_real = np.random.choice(len(real_depth_pts), np.min([len(real_depth_pts), len(synthetic_pts), N]))
        sub_idcs_syn = np.random.choice(len(synthetic_pts), np.min([len(real_depth_pts), len(synthetic_pts), N]))
        #T = icp_wrap(synthetic_pts, real_depth_pts, threshold=0.000000002)
        T, distances, iterations = icp(synthetic_pts[sub_idcs_syn], real_depth_pts[sub_idcs_real],
                                       tolerance=0.0000005, verbose=verbose, depth_only=depth_only, no_depth=no_depth)

        if no_depth == True:
            angle, _, _ = transform.rotation_from_matrix(T)
            if np.abs(angle) > angle_change_limit:
                T = np.eye(4)

        H_est = np.zeros((4, 4))
        # R_est, t_est is from model to camera
        H_est[3, 3] = 1
        H_est[:3, 3] = t_est
        H_est[:3, :3] = R_est

        H_est_refined = np.dot(T, H_est)

        R_refined = H_est_refined[:3, :3]
        t_refined = H_est_refined[:3, 3]

    return (R_refined, t_refined)

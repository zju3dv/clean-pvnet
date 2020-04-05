from lib.utils.renderer import opengl_utils
import numpy as np
from lib.utils.pysixd import transform
from sklearn.neighbors import NearestNeighbors


def rgbd_to_point_cloud(K, depth):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    return pts


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
        # t = np.array([0, 0, t[2]])
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


def icp(A, B, init_pose=None, max_iterations=200, tolerance=0.001, verbose=False, depth_only=False, no_depth=False):
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


class ICPRefiner:
    def __init__(self, model, im_size):
        self.renderer = opengl_utils.DepthRender(model, im_size)
        self.im_size = im_size

    def refine(self, depth_crop, R_est, t_est, K_test, depth_only=False, no_depth=False, max_mean_dist_factor=2.0):

        depth = self.renderer.render(self.im_size, 100, 10000, K_test, R_est, t_est)
        synthetic_pts = rgbd_to_point_cloud(K_test, depth)

        centroid_synthetic_pts = np.mean(synthetic_pts, axis=0)
        try:
            max_mean_dist = np.max(np.linalg.norm(synthetic_pts - centroid_synthetic_pts, axis=1))
        except:
            return (R_est, t_est)

        real_depth_pts = rgbd_to_point_cloud(K_test, depth_crop)

        real_synmean_dist = np.linalg.norm(real_depth_pts - centroid_synthetic_pts, axis=1)
        real_depth_pts = real_depth_pts[real_synmean_dist < max_mean_dist_factor * max_mean_dist]
        if len(real_depth_pts) < len(synthetic_pts) / 20.:
            print('not enough visible points')
            R_refined = R_est
            t_refined = t_est
        else:
            N = 3000
            sub_idcs_real = np.random.choice(len(real_depth_pts), np.min([len(real_depth_pts), len(synthetic_pts), N]))
            sub_idcs_syn = np.random.choice(len(synthetic_pts), np.min([len(real_depth_pts), len(synthetic_pts), N]))
            T, distances, iterations = icp(synthetic_pts[sub_idcs_syn], real_depth_pts[sub_idcs_real], tolerance=0.0000005, depth_only=depth_only, no_depth=no_depth)

            if no_depth == True:
                angle, _, _ = transform.rotation_from_matrix(T)
                angle_change_limit = 20 * np.pi / 180.
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

        return R_refined, t_refined

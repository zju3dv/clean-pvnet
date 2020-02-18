import numpy as np
from lib.config import cfg

mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835],
               dtype=np.float32).reshape(1, 1, 3)
data_rng = np.random.RandomState(123)
eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                   dtype=np.float32)
eig_vec = np.array([
    [-0.58752847, -0.69563484, 0.41340352],
    [-0.5832747, 0.00994535, -0.81221408],
    [-0.56089297, 0.71832671, 0.41158938]
], dtype=np.float32)


down_ratio = 4

num_obj_in_training_image = 6
train_w, train_h = 720, 540

ct_score = 0.2

visib_gt_min = 0.1
vsd_cost = 'step'
vsd_delta = 15
vsd_tau = 20
error_thresh_vsd = 0.3

pvnet_input_scale = cfg.tless.pvnet_input_scale
scale_train_ratio = cfg.tless.scale_train_ratio
scale_ratio = cfg.tless.scale_ratio
box_train_ratio = cfg.tless.box_train_ratio
box_ratio = cfg.tless.box_ratio

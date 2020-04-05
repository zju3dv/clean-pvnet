from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
if cfg.test.icp:
    from lib.utils.icp import icp_utils
    # from lib.utils.icp.icp_refiner.build import ext_
if cfg.test.un_pnp:
    from lib.csrc.uncertainty_pnp import un_pnp_utils
    import scipy
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
from lib.utils.vsd import inout
from transforms3d.quaternions import mat2quat, quat2mat
from lib.csrc.nn import nn_utils


class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = os.path.join(result_dir, cfg.test.dataset)
        os.system('mkdir -p {}'.format(self.result_dir))

        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        data_root = args['data_root']
        cls = cfg.cls_type
        model_path = os.path.join('data/linemod', cls, cls + '.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        self.diameter = linemod_config.diameters[cls] / 100

        self.proj2d = []
        self.add = []
        self.cmd5 = []

        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []

        self.mask_ap = []

        self.height = 480
        self.width = 640

        model = inout.load_ply(model_path)
        model['pts'] = model['pts'] * 1000
        self.icp_refiner = icp_utils.ICPRefiner(model, (self.width, self.height)) if cfg.test.icp else None
        # if cfg.test.icp:
        #     self.icp_refiner = ext_.Synthesizer(os.path.realpath(model_path))
        #     self.icp_refiner.setup(self.width, self.height)

    def projection_2d(self, pose_pred, pose_targets, K, icp=False, threshold=5):
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))
        if icp:
            self.icp_proj2d.append(proj_mean_diff < threshold)
        else:
            self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
            mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))

        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def cm_degree_5_metric(self, pose_pred, pose_targets, icp=False):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        if icp:
            self.icp_cmd5.append(translation_distance < 5 and angular_distance < 5)
        else:
            self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.7)

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)

        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

        return pose_pred

    def uncertainty_pnp(self, kpt_3d, kpt_2d, var, K):
        cov_invs = []
        for vi in range(var.shape[0]):
            if var[vi, 0, 0] < 1e-6 or np.sum(np.isnan(var)[vi]) > 0:
                cov_invs.append(np.zeros([2, 2]).astype(np.float32))
            else:
                cov_inv = np.linalg.inv(scipy.linalg.sqrtm(var[vi]))
                cov_invs.append(cov_inv)

        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]
        pose_pred = un_pnp_utils.uncertainty_pnp(kpt_2d, weights, kpt_3d, K)

        return pose_pred

    def icp_refine_(self, pose, anno, output):
        depth = read_depth(anno['depth_path']).astype(np.uint16)
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask = mask.astype(np.int32)
        pose = pose.astype(np.float32)

        poses = np.zeros([1, 7], dtype=np.float32)
        poses[0, :4] = mat2quat(pose[:, :3])
        poses[0, 4:] = pose[:, 3]

        poses_new = np.zeros([1, 7], dtype=np.float32)
        poses_icp = np.zeros([1, 7], dtype=np.float32)

        fx = 572.41140
        fy = 573.57043
        px = 325.26110
        py = 242.04899
        zfar = 6.0
        znear = 0.25;
        factor= 1000.0
        error_threshold = 0.01

        rois = np.zeros([1, 6], dtype=np.float32)
        rois[:, :] = 1

        self.icp_refiner.solveICP(mask, depth,
            self.height, self.width,
            fx, fy, px, py,
            znear, zfar,
            factor,
            rois.shape[0], rois,
            poses, poses_new, poses_icp,
            error_threshold
        )

        pose_icp = np.zeros([3, 4], dtype=np.float32)
        pose_icp[:, :3] = quat2mat(poses_icp[0, :4])
        pose_icp[:, 3] = poses_icp[0, 4:]

        return pose_icp

    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
        if cfg.test.un_pnp:
            var = output['var'][0].detach().cpu().numpy()
            pose_pred = self.uncertainty_pnp(kpt_3d, kpt_2d, var, K)
        else:
            pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        if cfg.test.icp:
            pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
            if cfg.cls_type in ['eggbox', 'glue']:
                self.add_metric(pose_pred_icp, pose_gt, syn=True, icp=True)
            else:
                self.add_metric(pose_pred_icp, pose_gt, icp=True)
            self.projection_2d(pose_pred_icp, pose_gt, K, icp=True)
            self.cm_degree_5_metric(pose_pred_icp, pose_gt, icp=True)

        if cfg.cls_type in ['eggbox', 'glue']:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
        self.projection_2d(pose_pred, pose_gt, K)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.mask_iou(output, batch)

    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        print('2d projections metric: {}'.format(proj2d))
        print('ADD metric: {}'.format(add))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        print('mask ap70: {}'.format(ap))
        if cfg.test.icp:
            print('2d projections metric after icp: {}'.format(np.mean(self.icp_proj2d)))
            print('ADD metric after icp: {}'.format(np.mean(self.icp_add)))
            print('5 cm 5 degree metric after icp: {}'.format(np.mean(self.icp_cmd5)))
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_proj2d = []
        self.icp_add = []
        self.icp_cmd5 = []
        return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap}

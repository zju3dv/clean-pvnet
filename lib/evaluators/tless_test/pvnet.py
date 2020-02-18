import os
import pycocotools.mask as mask_util
import pycocotools.coco as coco
from lib.utils import data_utils
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
from lib.utils.tless import tless_config
import json
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils
from lib.utils.vsd import inout
from PIL import Image
from lib.csrc.nn import nn_utils
import yaml
if cfg.test.un_pnp:
    from lib.csrc.uncertainty_pnp import un_pnp_utils
    import scipy
if cfg.test.icp or cfg.test.vsd:
    from lib.utils.icp import icp_utils
    # from lib.utils.icp.icp_refiner.build import ext_
import torch
import cv2
from transforms3d.quaternions import mat2quat, quat2mat
import tqdm


class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = result_dir
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.obj_id = int(args['obj_id'])
        self.coco = coco.COCO(self.ann_file)
        self.gt_img_ids = self.coco.getImgIds(catIds=[self.obj_id])

        model_dir = 'data/tless/models_cad'
        obj_path = os.path.join(model_dir, 'obj_{:02d}.ply'.format(self.obj_id))
        self.model = inout.load_ply(obj_path)
        self.model_pts = self.model['pts'] / 1000.

        model_info = yaml.load(open(os.path.join(model_dir, 'models_info.yml')))
        self.diameter = model_info[self.obj_id]['diameter'] / 1000.

        self.vsd = []
        self.adi = []
        self.cmd5 = []

        self.icp_vsd = []
        self.icp_adi = []
        self.icp_cmd5 = []

        self.pose_per_id = []
        self.pose_icp_per_id = []
        self.img_ids = []

        self.height = 540
        self.width = 720

        if cfg.test.icp or cfg.test.vsd:
            self.icp_refiner = icp_utils.ICPRefiner(self.model, (self.width, self.height))
            # model_path = os.path.join(model_dir, 'colobj_{:02d}.ply'.format(self.obj_id))
            # self.icp_refiner = ext_.Synthesizer(os.path.realpath(model_path))
            # self.icp_refiner.setup(self.width, self.height)

    def vsd_metric(self, pose_pred, pose_gt, K, depth_path, icp=False):
        from lib.utils.vsd import vsd_utils

        depth = inout.load_depth(depth_path) * 0.1
        im_size = (depth.shape[1], depth.shape[0])
        dist_test = vsd_utils.misc.depth_im_to_dist_im(depth, K)

        delta = tless_config.vsd_delta
        tau = tless_config.vsd_tau
        cost_type = tless_config.vsd_cost
        error_thresh = tless_config.error_thresh_vsd

        depth_gt = {}
        dist_gt = {}
        visib_gt = {}

        for pose_pred_ in pose_pred:
            R_est = pose_pred_[:, :3]
            t_est = pose_pred_[:, 3:] * 1000
            depth_est = self.icp_refiner.renderer.render(im_size, 100, 10000, K, R_est, t_est)
            # depth_est = self.opengl.render(im_size, 100, 10000, K, R_est, t_est)
            dist_est = vsd_utils.misc.depth_im_to_dist_im(depth_est, K)

            for gt_id, pose_gt_ in enumerate(pose_gt):
                R_gt = pose_gt_[:, :3]
                t_gt = pose_gt_[:, 3:] * 1000
                if gt_id not in visib_gt:
                    depth_gt_ = self.icp_refiner.renderer.render(im_size, 100, 10000, K, R_gt, t_gt)
                    # depth_gt_ = self.opengl.render(im_size, 100, 10000, K, R_gt, t_gt)
                    dist_gt_ = vsd_utils.misc.depth_im_to_dist_im(depth_gt_, K)
                    dist_gt[gt_id] = dist_gt_
                    visib_gt[gt_id] = vsd_utils.visibility.estimate_visib_mask_gt(
                        dist_test, dist_gt_, delta)

                e = vsd_utils.vsd(dist_est, dist_gt[gt_id], dist_test, visib_gt[gt_id],
                                  delta, tau, cost_type)
                if e < error_thresh:
                    return 1

        return 0

    def adi_metric(self, pose_pred, pose_gt, percentage=0.1):
        diameter = self.diameter * percentage
        for pose_pred_ in pose_pred:
            for pose_gt_ in pose_gt:
                model_pred = np.dot(self.model_pts, pose_pred_[:, :3].T) + pose_pred_[:, 3]
                model_targets = np.dot(self.model_pts, pose_gt_[:, :3].T) + pose_gt_[:, 3]
                idxs = nn_utils.find_nearest_point_idx(model_pred, model_targets)
                mean_dist = np.mean(np.linalg.norm(model_pred[idxs] - model_targets, 2, 1))
                if mean_dist < diameter:
                    return 1
        return 0

    def cm_degree_5_metric(self, pose_pred, pose_gt):
        for pose_pred_ in pose_pred:
            for pose_gt_ in pose_gt:
                trans_distance, ang_distance = pvnet_pose_utils.cm_degree_5(pose_pred_, pose_gt_)
                if trans_distance < 5 and ang_distance < 5:
                    return 1
        return 0

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

    def icp_refine(self, pose_pred, depth_path, mask, K):
        depth = inout.load_depth(depth_path).astype(np.int32) / 10.
        mask = mask.astype(np.int32)
        pose = pose_pred.astype(np.float32)

        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000

        R_refined, t_refined = self.icp_refiner.refine(depth, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), depth_only=True, max_mean_dist_factor=5.0)
        R_refined, _ = self.icp_refiner.refine(depth, R_refined, t_refined, K.copy(), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))

        return pose_pred

    def icp_refine_(self, pose, depth_path, mask, K):
        depth = inout.load_depth(depth_path).astype(np.uint16)
        mask = mask.astype(np.int32)
        pose = pose.astype(np.float32)

        box = cv2.boundingRect(mask.astype(np.uint8))
        x, y = box[0] + box[2] / 2., box[1] + box[3] / 2.
        z = np.mean(depth[mask != 0] / 10000.)
        x = ((x - K[0, 2]) * z) / float(K[0, 0])
        y = ((y - K[1, 2]) * z) / float(K[1, 1])
        center = [x, y, z]
        pose[:, 3] = center

        poses = np.zeros([1, 7], dtype=np.float32)
        poses[0, :4] = mat2quat(pose[:, :3])
        poses[0, 4:] = pose[:, 3]

        poses_new = np.zeros([1, 7], dtype=np.float32)
        poses_icp = np.zeros([1, 7], dtype=np.float32)

        fx = K[0, 0]
        fy = K[1, 1]
        px = K[0, 2]
        py = K[1, 2]
        zfar = 6.0
        znear = 0.25
        factor = 10000.
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
        img_id = int(batch['meta']['img_id'])
        self.img_ids.append(img_id)
        img_data = self.coco.loadImgs(int(img_id))[0]
        depth_path = img_data['depth_path']

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.obj_id)
        annos = self.coco.loadAnns(ann_ids)
        kpt_3d = np.concatenate([annos[0]['fps_3d'], [annos[0]['center_3d']]], axis=0)
        corner_3d = np.array(annos[0]['corner_3d'])
        K = np.array(annos[0]['K'])
        pose_gt = [np.array(anno['pose']) for anno in annos]

        kpt_2d = output['kpt_2d'].detach().cpu().numpy()
        centers = batch['meta']['center']
        scales = batch['meta']['scale']
        boxes = batch['meta']['box']
        h, w = batch['inp'].size(2), batch['inp'].size(3)

        pose_preds = []
        pose_preds_icp = []
        for i in range(len(centers)):
            center = centers[i].detach().cpu().numpy()
            scale = scales[i].detach().cpu().numpy()
            kpt_2d_ = kpt_2d[i]
            trans_inv = data_utils.get_affine_transform(center[0], scale[0], 0, [w, h], inv=1)
            kpt_2d_ = data_utils.affine_transform(kpt_2d_, trans_inv)
            if cfg.test.un_pnp:
                var = output['var'][i].detach().cpu().numpy()
                pose_pred = self.uncertainty_pnp(kpt_3d, kpt_2d_, var, K)
            else:
                pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_, K)
            pose_preds.append(pose_pred)

            if cfg.test.icp:
                seg = torch.argmax(output['seg'][i], dim=0).detach().cpu().numpy()
                seg = seg.astype(np.uint8)
                seg = cv2.warpAffine(seg, trans_inv, (self.width, self.height), flags=cv2.INTER_NEAREST)
                pose_pred_icp = self.icp_refine(pose_pred.copy(), depth_path, seg.copy(), K.copy())
                pose_preds_icp.append(pose_pred_icp)

        if cfg.test.icp:
            self.icp_adi.append(self.adi_metric(pose_preds_icp, pose_gt))
            self.icp_cmd5.append(self.cm_degree_5_metric(pose_preds_icp, pose_gt))
            self.pose_icp_per_id.append(pose_preds_icp)

        self.adi.append(self.adi_metric(pose_preds, pose_gt))
        self.cmd5.append(self.cm_degree_5_metric(pose_preds, pose_gt))
        self.pose_per_id.append(pose_preds)

    def summarize_vsd(self, pose_preds, img_ids, vsd):
        for pose_pred, img_id in tqdm.tqdm(zip(pose_preds, img_ids)):
            img_data = self.coco.loadImgs(int(img_id))[0]
            depth_path = img_data['depth_path']

            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.obj_id)
            annos = self.coco.loadAnns(ann_ids)
            K = np.array(annos[0]['K'])
            pose_gt = [np.array(anno['pose']) for anno in annos]
            vsd.append(self.vsd_metric(pose_pred, pose_gt, K, depth_path))

    def summarize(self):
        if cfg.test.vsd:
            from lib.utils.vsd import vsd_utils
            # self.opengl = vsd_utils.renderer.DepthRender(self.model, (720, 540))
            self.summarize_vsd(self.pose_per_id, self.img_ids, self.vsd)
            if cfg.test.icp:
                self.summarize_vsd(self.pose_icp_per_id, self.img_ids, self.icp_vsd)
            self.pose_per_id = []
            self.pose_icp_per_id = []
            self.img_ids = []

        vsd = np.sum(self.vsd) / len(self.gt_img_ids)
        adi = np.sum(self.adi) / len(self.gt_img_ids)
        cmd5 = np.sum(self.cmd5) / len(self.gt_img_ids)
        icp_vsd = np.sum(self.icp_vsd) / len(self.gt_img_ids)
        icp_adi = np.sum(self.icp_adi) / len(self.gt_img_ids)
        icp_cmd5 = np.sum(self.icp_cmd5) / len(self.gt_img_ids)

        print('vsd metric: {}'.format(vsd))
        print('adi metric: {}'.format(adi))
        print('5 cm 5 degree metric: {}'.format(cmd5))
        if cfg.test.icp:
            print('vsd metric after icp: {}'.format(icp_vsd))
            print('adi metric after icp: {}'.format(icp_adi))
            print('5 cm 5 degree metric after icp: {}'.format(icp_cmd5))

        self.vsd = []
        self.adi = []
        self.cmd5 = []
        self.icp_vsd = []
        self.icp_adi = []
        self.icp_cmd5 = []

        return {'vsd': vsd, 'adi': adi, 'cmd5': cmd5}

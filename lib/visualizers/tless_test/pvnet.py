from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
from PIL import Image
from lib.utils import data_utils
import cv2
import torch


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.obj_id = int(args['obj_id'])
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        img_id = int(batch['meta']['img_id'])
        img_data = self.coco.loadImgs(int(img_id))[0]
        path = img_data['file_name']
        depth_path = img_data['depth_path']
        img = np.array(Image.open(path))

        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.obj_id)
        annos = self.coco.loadAnns(ann_ids)
        kpt_3d = np.concatenate([annos[0]['fps_3d'], [annos[0]['center_3d']]], axis=0)
        corner_3d = np.array(annos[0]['corner_3d'])
        K = np.array(annos[0]['K'])

        kpt_2d = output['kpt_2d'].detach().cpu().numpy()
        centers = batch['meta']['center']
        scales = batch['meta']['scale']
        boxes = batch['meta']['box']
        h, w = batch['inp'].size(2), batch['inp'].size(3)

        kpt_2ds = []
        segs = []
        for i in range(len(centers)):
            center = centers[i].detach().cpu().numpy()
            scale = scales[i].detach().cpu().numpy()
            kpt_2d_ = kpt_2d[i]
            trans_inv = data_utils.get_affine_transform(center[0], scale[0], 0, [w, h], inv=1)
            kpt_2d_ = data_utils.affine_transform(kpt_2d_, trans_inv)
            kpt_2ds.append(kpt_2d_)

            seg = torch.argmax(output['seg'][i], dim=0).detach().cpu().numpy()
            seg = seg.astype(np.uint8)
            seg = cv2.warpAffine(seg, trans_inv, (720, 540), flags=cv2.INTER_NEAREST)
            segs.append(seg)

        _, ax = plt.subplots(1)
        ax.imshow(img)

        # for i in range(len(boxes)):
        #     x_min, y_min, x_max, y_max = boxes[i].view(-1).numpy()
        #     ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])

        depth = np.array(Image.open(depth_path)).astype(np.float32)

        for i, kpt_2d in enumerate(kpt_2ds):
            pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

            mask = segs[i]
            box = cv2.boundingRect(mask.astype(np.uint8))
            x, y = box[0] + box[2] / 2., box[1] + box[3] / 2.
            z = np.mean(depth[mask != 0] / 10000.)
            x = ((x - K[0, 2]) * z) / float(K[0, 0])
            y = ((y - K[1, 2]) * z) / float(K[1, 1])
            center = [x, y, z]

            # pose_pred[:, 3] = center

            corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
            ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
            ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))

        for anno in annos:
            pose_gt = np.array(anno['pose'])
            corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
            ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
            ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))

        plt.show()


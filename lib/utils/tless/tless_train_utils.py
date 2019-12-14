from lib.utils.tless.tless_utils import *
from lib.utils.tless import tless_utils, tless_config
from PIL import Image
import numpy as np
from imgaug.augmenters import *


_data_rng = tless_config.data_rng
_eig_val = tless_config.eig_val
_eig_vec = tless_config.eig_vec
mean = tless_config.mean
std = tless_config.std


crop_scale = np.array([512, 416])
input_scale = np.array([512, 416])


augmenters = Sequential([
    Sometimes(0.2, GaussianBlur(0.4)),
    Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
    Sometimes(0.4, Add((-15, 15), per_channel=0.5)),
    # Sometimes(0.3, Invert(0.2, per_channel=True)),
    Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
    # Sometimes(0.5, Multiply((0.6, 1.4))),
    Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3)),
    # Sometimes(0.2, CoarseDropout( p=0.1, size_px = 10, size_percent=0.001) )
], random_order=True)


color_jitter = Sequential([
    Sometimes(0.5, Add((-15, 15), per_channel=0.5)),
    Sometimes(0.5, AddToHueAndSaturation((-20, 20))),
    Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))
])


def augment(img, bboxes, split):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = crop_scale
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    if split == 'train':
        scale = scale * np.random.uniform(1., 2.)

        bbox = bboxes[np.random.randint(0, len(bboxes))]
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        border_r = max(width - border, border + 1)
        center[0] = np.clip(center[0], a_min=border, a_max=border_r)

        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        border_r = max(height - border, border + 1)
        center[1] = np.clip(center[1], a_min=border, a_max=border_r)
        input_w, input_h = input_scale

        rot = np.random.uniform(-60, 60)

    if split != 'train':
        center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        scale = max(width, height) * 1.0
        scale = np.array([scale, scale])
        x = 32
        input_w, input_h = int((width / 1. + x - 1) // x * x), int((height / 1. + x - 1) // x * x)
        rot = 0

    trans_input = data_utils.get_affine_transform(
        center, scale, rot, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation
    inp = augmenters.augment_image(inp)
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // tless_config.down_ratio, input_w // tless_config.down_ratio
    trans_output = data_utils.get_affine_transform(
        center, scale, rot, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, trans_input, trans_output, center, scale, inp_out_hw


def cut_and_paste(bg_paths, rgb_paths, mask_paths, category_ids):
    if np.random.uniform() < 0.9:
        train_img = cv2.imread(bg_paths[np.random.randint(len(bg_paths))])
        train_img = cv2.resize(
            train_img, (tless_config.train_w, tless_config.train_h))
    else:
        train_img = np.zeros(
            (tless_config.train_h, tless_config.train_w, 3), dtype=np.uint8)
    train_mask = np.zeros(
        (tless_config.train_h, tless_config.train_w), dtype=np.int16)

    for instance_id in range(len(rgb_paths)):
        rgb_path = rgb_paths[instance_id]
        mask_path = mask_paths[instance_id]
        category_id = category_ids[instance_id]

        img = cv2.imread(rgb_path)
        mask = np.array(Image.open(mask_path))

        rot = np.random.uniform() * 360
        img = rotate_image(img, rot)
        mask = rotate_image(mask, rot)

        mask_id = category_id * 1000 + instance_id
        tless_utils.cut_and_paste(img, mask, train_img, train_mask, mask_id)

    return train_img, train_mask


def get_fused_image(coco_db, ann_ids, bg_paths):
    img_ids = np.random.choice(ann_ids, tless_config.num_obj_in_training_image)
    rgb_paths = []
    mask_paths = []
    category_ids = []
    for instance_id, img_id in enumerate(img_ids):
        ann_ids = coco_db.getAnnIds(imgIds=img_id)
        anno = coco_db.loadAnns(ann_ids)[0]
        rgb_paths.append(coco_db.loadImgs(int(img_id))[0]['rgb_path'])
        mask_paths.append(anno['mask_path'])
        category_ids.append(1)
    img, mask = cut_and_paste(bg_paths, rgb_paths, mask_paths, category_ids)
    mask = (mask != 0).astype(np.uint8)
    return img, mask


def get_bbox(mask, category_ids):
    height, width = mask.shape[0], mask.shape[1]
    bboxes = []
    for instance_id in range(len(category_ids)):
        category_id = category_ids[instance_id]
        mask_id = category_id * 1000 + instance_id
        mask_ = (mask == mask_id).astype(np.uint8)
        bbox = tless_utils.xywh_to_xyxy(cv2.boundingRect(mask_))
        bbox[2] = min(bbox[2], width - 1)
        bbox[3] = min(bbox[3], height - 1)
        bboxes.append(bbox)
    return bboxes


def rotate_image(mat, angle, get_rot=False):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h), cv2.INTER_NEAREST)

    if get_rot == False:
        return rotated_mat
    else:
        return rotated_mat, rotation_mat

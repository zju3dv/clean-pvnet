from lib.utils.tless import tless_config
from lib.utils import data_utils
import numpy as np
import cv2


_data_rng = tless_config.data_rng
_eig_val = tless_config.eig_val
_eig_vec = tless_config.eig_vec
mean = tless_config.mean
std = tless_config.std


crop_scale = np.array([512, 416])
input_scale = np.array([512, 416])


def augment(img, split):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(img.shape[0], img.shape[1]) * 1.0
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    if split == 'train':
        scale = scale * np.random.uniform(0.6, 1.4)
        center = np.array([0, 0])

        border = scale[0] // 2 if scale[0] < width else width - scale[0] // 2
        border_r = max(width - border, border + 1)
        center[0] = np.random.randint(border, border_r)

        border = scale[1] // 2 if scale[1] < height else height - scale[1] // 2
        border_r = max(height - border, border + 1)
        center[1] = np.random.randint(border, border_r)
        input_w, input_h = input_scale

    if split != 'train':
        center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        scale = max(width, height) * 1.0
        scale = np.array([scale, scale])
        x = 32
        input_w, input_h = (width + x - 1) // x * x, (height + x - 1) // x * x

    trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    # color augmentation
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    output_h, output_w = input_h // tless_config.down_ratio, input_w // tless_config.down_ratio
    trans_output = data_utils.get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)

    return orig_img, inp, trans_input, trans_output, center, scale, inp_out_hw


def xywh_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]


def transform_bbox(bboxes, trans_output, h, w):
    new_bboxes = []
    for i in range(len(bboxes)):
        box = np.array(bboxes[i]).reshape(-1, 2)
        box = data_utils.affine_transform(box, trans_output)
        box[:, 0] = np.clip(box[:, 0], 0, w-1)
        box[:, 1] = np.clip(box[:, 1], 0, h-1)
        box = box.ravel().tolist()

        if box[2] - box[0] <= 1 or box[3] - box[1] <= 1:
            new_bboxes.append([])
        else:
            new_bboxes.append(box)
    return new_bboxes


def cut_and_paste(img, mask, train_img, train_mask, mask_id):
    ys, xs = np.nonzero(mask)
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    h, w = y_max - y_min, x_max - x_min
    img_h, img_w = train_img.shape[0], train_img.shape[1]

    dst_y, dst_x = np.random.randint(0, img_h-h), np.random.randint(0, img_w-w)
    dst_ys, dst_xs = ys - y_min + dst_y, xs - x_min + dst_x

    train_img[dst_ys, dst_xs] = img[ys, xs]
    train_mask[dst_ys, dst_xs] = mask_id

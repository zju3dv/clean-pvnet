from .tless_train_utils import *


_data_rng = tless_config.data_rng
_eig_val = tless_config.eig_val
_eig_vec = tless_config.eig_vec
mean = tless_config.mean
std = tless_config.std

input_scale = np.array(tless_config.pvnet_input_scale)


def magnify_box(box, times, h, w):
    box_mean = np.mean(box, axis=0)
    box = np.round((box - box_mean) * times + box_mean).astype(np.int32)
    box[:, 0] = np.clip(box[:, 0], a_min=0, a_max=w-1)
    box[:, 1] = np.clip(box[:, 1], a_min=0, a_max=h-1)
    return box


def augment(img, bbox, split):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = max(bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    if split == 'train':
        scale_train_ratio = tless_config.scale_train_ratio
        scale = scale * np.random.uniform(scale_train_ratio[0], scale_train_ratio[1])

        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        center_pure = center.copy()

        # shift_x = 16
        # center[0] += np.random.uniform(scale[0]/shift_x, scale[0]/shift_x)
        # center[1] += np.random.uniform(scale[1]/shift_x, scale[1]/shift_x)
        b_w, b_h = scale[0], scale[1]

        input_w, input_h = input_scale

        rot = 0

    trans_input = data_utils.get_affine_transform(
        center, scale, rot, [input_w, input_h])
    trans_pure = data_utils.get_affine_transform(
        center_pure, scale, rot, [input_w, input_h])
    inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

    if split == 'train':
        box = np.array(bbox).reshape(-1, 2)
        box = data_utils.affine_transform(box, trans_pure)
        box_ratio = np.random.uniform(tless_config.box_train_ratio[0], tless_config.box_train_ratio[1])
        box = magnify_box(box, box_ratio, input_h, input_w)
        new_img = np.zeros_like(inp)
        new_img[box[0, 1]:box[1, 1]+1, box[0, 0]:box[1, 0]+1] = inp[box[0, 1]:box[1, 1]+1, box[0, 0]:box[1, 0]+1]
        inp = new_img

    # color augmentation
    inp = augmenters.augment_image(inp)
    orig_img = inp.copy()
    inp = (inp.astype(np.float32) / 255.)
    if split == 'train':
        data_utils.color_aug(_data_rng, inp, _eig_val, _eig_vec)

    # normalize the image
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)

    inp_hw = (input_h, input_w)

    return orig_img, inp, trans_input, center, scale, inp_hw

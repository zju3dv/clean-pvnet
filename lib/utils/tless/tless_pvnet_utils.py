from .tless_train_utils import *


_data_rng = tless_config.data_rng
_eig_val = tless_config.eig_val
_eig_vec = tless_config.eig_vec
mean = tless_config.mean
std = tless_config.std

input_scale = np.array([256, 256])


def augment(img, bbox, split):
    # resize input
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    scale = np.array([bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1], dtype=np.float32)
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    if split == 'train':
        scale = scale * np.random.uniform(0.8, 1.2)

        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        b_w, b_h = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
        center[0] = center[0] + np.random.uniform(-b_w/8, b_w/8)
        center[1] = center[1] + np.random.uniform(-b_h/8, b_h/8)

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

    inp_hw = (input_h, input_w)

    return orig_img, inp, trans_input, center, scale, inp_hw

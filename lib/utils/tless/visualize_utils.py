import matplotlib.pyplot as plt
from lib.utils.tless import tless_config
from lib.utils import img_utils
import numpy as np
import cv2


def visualize_bbox(img, bboxes):
    plt.imshow(img[..., [2, 1, 0]])
    for box in bboxes:
        x_min, y_min, x_max, y_max = box
        plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])
    plt.show()


def visualize_detection(img, data):

    def blend_hm_img(hm, img):
        hm = np.max(hm, axis=0)
        h, w = hm.shape[:2]
        img = cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        hm = np.array([255, 255, 255]) - (hm.reshape(h, w, 1) * img_utils.colors[0]).astype(np.uint8)
        ratio = 0.5
        blend = (img * ratio + hm * (1 - ratio)).astype(np.uint8)
        return blend

    img = img_utils.bgr_to_rgb(img)
    blend = blend_hm_img(data['ct_hm'], img)

    plt.imshow(blend)
    ct_ind = np.array(data['ct_ind'])
    w = img.shape[1] // tless_config.down_ratio
    xs = ct_ind % w
    ys = ct_ind // w
    for i in range(len(data['wh'])):
        w, h = data['wh'][i]
        x_min, y_min = xs[i] - w / 2, ys[i] - h / 2
        x_max, y_max = xs[i] + w / 2, ys[i] + h / 2
        plt.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min])
    plt.show()


def visualize_img(orig_imgs):
    plt.imshow(orig_imgs[0][..., [2, 1, 0]])
    plt.show()

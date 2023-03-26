"""
augmentation模块
==================

本模块实现了一些数据增强(主要是几何变换)的常用函数
"""

# 标准库
import math

# 第三方库
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def resize_keep_aspect_ratio(img, imsize, intp_type=cv2.INTER_LINEAR):
    h, w = img.shape[0], img.shape[1]
    ratio = imsize / max(h, w)
    hbeg, wbeg = 0, 0
    # padding_mask=np.zeros([imsize,imsize],np.uint8)
    if h > w:
        hnew = imsize
        wnew = int(ratio * w)
        img = cv2.resize(img, (wnew, hnew), interpolation=intp_type)
        if wnew < imsize:
            if len(img.shape) == 3:
                img_pad = np.zeros([imsize, imsize, img.shape[2]], img.dtype)
            else:
                img_pad = np.zeros([imsize, imsize], img.dtype)
            wbeg = int((imsize - wnew) / 2)
            img_pad[:, wbeg:wbeg + wnew] = img

            # padding_mask[:,:wbeg]=1
            # padding_mask[:,wbeg+wnew:]=1
            img = img_pad
    else:
        hnew = int(ratio * h)
        wnew = imsize
        img = cv2.resize(img, (wnew, hnew), interpolation=intp_type)
        if hnew < imsize:
            if len(img.shape) == 3:
                img_pad = np.zeros([imsize, imsize, img.shape[2]], img.dtype)
            else:
                img_pad = np.zeros([imsize, imsize], img.dtype)
            hbeg = int((imsize - hnew) / 2)
            img_pad[hbeg:hbeg + hnew, :] = img

            # padding_mask[:,:hbeg]=1
            # padding_mask[:,hbeg+hnew:]=1
            img = img_pad

    # x_new=x_ori*ratio+wbeg
    # y_new=y_ori*ratio+hbeg
    return img, ratio, hbeg, wbeg


def rotate(img, mask, hcoords, rot_ang_min, rot_ang_max):
    """
    rotate _summary_

    :param img: _description_
    :type img: _type_
    :param mask: _description_
    :type mask: _type_
    :param hcoords: _description_
    :type hcoords: _type_
    :param rot_ang_min: _description_
    :type rot_ang_min: _type_
    :param rot_ang_max: _description_
    :type rot_ang_max: _type_
    :return: _description_
    :rtype: _type_
    """
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, R, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    last_row = np.asarray([[0, 0, 1]], np.float32)
    hcoords = np.matmul(hcoords, np.concatenate([R, last_row], 0).transpose())
    return img, mask, hcoords


def rotate_instance(img, mask, hcoords, rot_ang_min, rot_ang_max):
    """
    rotate_instance 以待检测目标中心为原点，随机选取角度degree∈[rot_ang_min, rot_ang_max)，对mask, img, hcoords进行旋转

    :param img: 输入图像
    :type img: narray
    :param mask: 图像掩码
    :type mask: narray
    :param hcoords: 2D关键点的齐次坐标
    :type hcoords: narray(9*3)
    :param rot_ang_min: 旋转角度的下限
    :type rot_ang_min: int
    :param rot_ang_max: 旋转角度的上限
    :type rot_ang_max: int
    :return: 图像，掩码，齐次坐标
    :rtype: tuple(img, mask, hcoords)
    """    
    h, w = img.shape[0], img.shape[1]
    degree = np.random.uniform(rot_ang_min, rot_ang_max)    # 随机选取旋转角度
    hs, ws = np.nonzero(mask)                               # 获取掩码中非零元素索引[hs,ws]
    R = cv2.getRotationMatrix2D((np.mean(ws), np.mean(hs)), degree, 1)  # 以目标中心为原点生成角度为degree的旋转矩阵
    mask = cv2.warpAffine(mask, R, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) # 旋转mask，最近点插值，边界填充为0/黑色
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) # 旋转img，线性插值，边界填充为0/黑色
    last_row = np.asarray([[0, 0, 1]], np.float32)
    hcoords = np.matmul(hcoords, np.concatenate([R, last_row], 0).transpose()) # 旋转特征点，等价于(R|1)hcoords^T
    return img, mask, hcoords


def flip(img, mask, hcoords):
    img = np.flip(img, 1)
    mask = np.flip(mask, 1)
    h, w = img.shape[0], img.shape[1]
    hcoords[:, 0] -= w / 2 * hcoords[:, 2]
    hcoords[:, 0] = -hcoords[:, 0]
    hcoords[:, 0] += w / 2 * hcoords[:, 2]
    return img, mask, hcoords


def crop_or_padding(img, mask, hcoords, hratio, wratio):
    '''
    if ratio<1.0 then crop, else padding
    :param img:
    :param mask:
    :param hcoords:
    :param hratio:
    :param wratio:
    :return:
    '''
    h, w, _ = img.shape
    hd = int(hratio * h - h)
    wd = int(wratio * w - w)
    hpad = hd > 0
    wpad = wd > 0

    if hpad:
        ohbeg = hd // 2
        ihbeg = 0
        hlen = h
    else:
        ohbeg = 0
        ihbeg = -hd // 2
        hlen = h + hd

    if wpad:
        owbeg = wd // 2
        iwbeg = 0
        wlen = w
    else:
        owbeg = 0
        iwbeg = -wd // 2
        wlen = w + wd

    out_img = np.zeros([h + hd, w + wd, 3], np.uint8)
    out_img[ohbeg:ohbeg + hlen, owbeg:owbeg + wlen] = img[ihbeg:ihbeg + hlen, iwbeg:iwbeg + wlen]
    out_mask = np.zeros([h + hd, w + wd], np.uint8)
    out_mask[ohbeg:ohbeg + hlen, owbeg:owbeg + wlen] = mask[ihbeg:ihbeg + hlen, iwbeg:iwbeg + wlen]
    hcoords[:, 1] -= hd * hcoords[:, 2]
    hcoords[:, 0] -= wd * hcoords[:, 2]

    return out_img, out_mask, hcoords,


def crop_or_padding_to_fixed_size_instance(img, mask, hcoords, th, tw, overlap_ratio=0.5):
    """
    crop_or_padding_to_fixed_size_instance 将输入图像(img)剪裁(crop)或填充(padding)至指定大小(th, tw)，
    图像中存在目标实例。

    :param img: 输入图像
    :type img: narray
    :param mask: 掩码
    :type mask: narray
    :param hcoords: 齐次坐标
    :type hcoords: narray
    :param th: 输出图像/掩码的高度
    :type th: int
    :param tw: 输出图像/掩码的宽度
    :type tw: int
    :param overlap_ratio: 重叠率(输出图像中目标区域相较于原区域的最小占比), 默认值为0.5
    :type overlap_ratio: float
    :return: 图像，掩码，齐次坐标
    :rtype: tuple(img, mask, hcoords)

    .. note:: 1. 该函数不需要保证对图像长度和宽度方向同时进行剪裁和填充(即可指定长度方向为剪裁，宽度方向为填充，反之亦然)。
              2. 原函数计算起始剪裁位置取值范围的公式有误，进行了修改。

    .. warning:: 当使用该函数剪裁图像时，需保证``th(tw) > overlap_ratio*fh(fw)``(fh，fw为原图像中目标区域的高度和宽度),
                 否则会使函数``numpy.random.randint(low,high)``报错(因为low>=hogh)。
    """
    h, w, _ = img.shape
    hs, ws = np.nonzero(mask)   # numpy.nonzero返回数组非零元素的下标(以维度分组)，与numpy.argwhere区分

    hmin, hmax = np.min(hs), np.max(hs)
    wmin, wmax = np.min(ws), np.max(ws)
    fh, fw = hmax - hmin, wmax - wmin
    hpad, wpad = th >= h, tw >= w       # 若目标(输出)图像的高/宽>=原图像的高/宽，则说明要对图像进行填充，置填充标志为True

    """ 修改 
    hrmax = int(min(hmin + overlap_ratio * fh, h - th))  # h must > target_height else hrmax<0
    hrmin = int(max(hmin + overlap_ratio * fh - th, 0))
    wrmax = int(min(wmin + overlap_ratio * fw, w - tw))  # w must > target_width else wrmax<0
    wrmin = int(max(wmin + overlap_ratio * fw - tw, 0))
    """
    # 确定剪裁起始位置的取值范围，以保证剪裁后的图像至少包含原目标区域的overlap_ratio的大小的部分(警告：使用时需保证th>overlap_ratio*fh，否则会出现hrmin>=hrmax)
    hrmax = int(min(hmax - overlap_ratio * fh, h - th))
    hrmin = int(max(hmin + overlap_ratio * fh - th, 0))
    wrmax = int(min(wmax - overlap_ratio * fw, w - tw))
    wrmin = int(max(wmin + overlap_ratio * fw - tw, 0))

    # 随机确定起始剪裁位置(若是填充图片，则起始位置为0)
    hbeg = 0 if hpad else np.random.randint(hrmin, hrmax)
    hend = hbeg + th
    wbeg = 0 if wpad else np.random.randint(wrmin,
                                            wrmax)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    wend = wbeg + tw

    # 获得剪裁后的img, mask
    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]

    # 获得关键点在剪裁图像上的齐次坐标
    hcoords[:, 0] -= wbeg * hcoords[:, 2]
    hcoords[:, 1] -= hbeg * hcoords[:, 2]

    # 若是填充图片(增大Δx)，则分别在图像两端填充长度为Δx/2的零
    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask
        hcoords[:, 0] += wbeg * hcoords[:, 2]
        hcoords[:, 1] += hbeg * hcoords[:, 2]

        img, mask = new_img, new_mask

    return img, mask, hcoords


def crop_or_padding_to_fixed_size(img, mask, th, tw):
    """
    crop_or_padding_to_fixed_size 将图像剪裁或填充至指定大小，图像中不存在目标实例。

    :param img: 输入图像
    :type img: narray
    :param mask: 掩码
    :type mask: narray
    :param th: 输出图像的高度
    :type th: int
    :param tw: 输出图像的宽度
    :type tw: int
    :return: 图像，掩码
    :rtype: tuple(img, mask)
    """
    h, w, _ = img.shape
    hpad, wpad = th >= h, tw >= w

    hbeg = 0 if hpad else np.random.randint(0, h - th)
    wbeg = 0 if wpad else np.random.randint(0,
                                            w - tw)  # if pad then [0,wend] will larger than [0,w], indexing it is safe
    hend = hbeg + th
    wend = wbeg + tw

    img = img[hbeg:hend, wbeg:wend]
    mask = mask[hbeg:hend, wbeg:wend]

    if hpad or wpad:
        nh, nw, _ = img.shape
        new_img = np.zeros([th, tw, 3], dtype=img.dtype)
        new_mask = np.zeros([th, tw], dtype=mask.dtype)

        hbeg = 0 if not hpad else (th - h) // 2
        wbeg = 0 if not wpad else (tw - w) // 2

        new_img[hbeg:hbeg + nh, wbeg:wbeg + nw] = img
        new_mask[hbeg:hbeg + nh, wbeg:wbeg + nw] = mask

        img, mask = new_img, new_mask

    return img, mask


def mask_out_instance(img, mask, min_side=0.1, max_side=0.3):
    ys, xs = np.nonzero(mask)
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    xlen = xmax - xmin
    ylen = ymax - ymin

    x_side = int(xlen * np.random.uniform(min_side, max_side) / 2)
    y_side = int(ylen * np.random.uniform(min_side, max_side) / 2)
    x_loc = np.random.randint(xmin, xmax)
    y_loc = np.random.randint(ymin, ymax)

    img[y_loc - y_side:y_loc + y_side, x_loc - x_side:x_loc + x_side] = np.random.randint(
        0, 255, img[y_loc - y_side:y_loc + y_side, x_loc - x_side:x_loc + x_side].shape)
    mask[y_loc - y_side:y_loc + y_side, x_loc - x_side:x_loc + x_side] = 0
    return img, mask


def blur_image(img, sigma=3):
    return cv2.GaussianBlur(img, (sigma, sigma), 0)


def add_noise(image):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.9:
        row, col, ch = image.shape
        mean = 0
        var = np.random.rand(1) * 0.3 * 256
        sigma = var ** 0.5
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size - 1) / 2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)
    return noisy


def compute_resize_range(mask, hmin, hmax, wmin, wmax):
    # compute resize
    ys, xs = np.nonzero(mask)
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    xlen = xmax - xmin
    ylen = ymax - ymin

    rmin, rmax = wmin / xlen, wmax / xlen
    rmax = min(rmax, hmax / ylen)
    rmin = max(rmin, hmin / ylen)

    return rmin, rmax


#### higher level api #####
def crop_resize_instance_v1(img, mask, hcoords, imheight, imwidth,
                            overlap_ratio=0.5, ratio_min=0.8, ratio_max=1.2):
    """
    crop_resize_instance_v1 将图像缩放至指定大小(imheight, imweidth)，
    同时随机选取比率ratio∈[ratio_min, ratio_max)对图像内容进行剪裁或填充。
    crop a region with [imheight*resize_ratio,imwidth*resize_ratio]
    which at least overlap with foreground bbox with overlap

    :param img: 输入图像
    :type img: narray
    :param mask: 掩码
    :type mask: narray
    :param hcoords: 齐次坐标
    :type hcoords: narray
    :param imheight: 输出图像的高度
    :type imheight: int
    :param imwidth: 输出图像的宽度
    :type imwidth: int
    :param overlap_ratio: 重叠率, 默认值为0.5
    :type overlap_ratio: float
    :param ratio_min: 最小剪裁率, 默认值为0.8
    :type ratio_min: float
    :param ratio_max: 最大填充率, 默认值1.2
    :type ratio_max: float
    :return: 图像，掩码，齐次坐标
    :rtype: tuple(img, mask, hcoords)
    """    

    # 确定图像剪裁或填充的大小
    resize_ratio = np.random.uniform(ratio_min, ratio_max)  # 随机选取剪裁(填充)率(resize_ratio)∈[ratio_min, ratio_max)
    target_height = int(imheight * resize_ratio)
    target_width = int(imwidth * resize_ratio)

    # 对图像内容进行剪裁或填充
    img, mask, hcoords = crop_or_padding_to_fixed_size_instance(
        img, mask, hcoords, target_height, target_width, overlap_ratio)

    # 将图像放大或缩小至(imwidth,imheight)，线性插值
    img = cv2.resize(img, (imwidth, imheight), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (imwidth, imheight), interpolation=cv2.INTER_NEAREST)

    # 调整齐次坐标
    hcoords[:, 0] = hcoords[:, 0] / resize_ratio
    hcoords[:, 1] = hcoords[:, 1] / resize_ratio

    return img, mask, hcoords


def crop_resize_instance_v2(img, mask, hcoords, imheight, imwidth,
                            overlap_ratio=0.5, hmin=30, hmax=135, wmin=30, wmax=130):
    '''

    crop a region with [imheight*resize_ratio,imwidth*resize_ratio]
    which at least overlap with foreground bbox with overlap
    :param img:
    :param mask:
    :param hcoords:
    :param imheight:
    :param imwidth:
    :param overlap_ratio:
    :param resize_ratio_min:
    :param resize_ratio_max:
    :return:
    '''
    if np.random.random() < 0.8:
        rmin, rmax = compute_resize_range(mask, hmin, hmax, wmin, wmax)
        resize_ratio = np.random.uniform(rmin, rmax)

        h, w = mask.shape
        target_height = int(h * resize_ratio)
        target_width = int(w * resize_ratio)
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        hcoords[:, 0] = hcoords[:, 0] * resize_ratio
        hcoords[:, 1] = hcoords[:, 1] * resize_ratio

    img, mask, hcoords = crop_or_padding_to_fixed_size_instance(
        img, mask, hcoords, imheight, imwidth, overlap_ratio)

    return img, mask, hcoords


def resize_with_crop_or_pad_to_fixed_size(img, mask, hcoords, ratio):
    h, w, _ = img.shape
    th, tw = int(math.ceil(h * ratio)), int(math.ceil(w * ratio))
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    hcoords[:, :2] *= ratio

    if ratio > 1.0:
        # crop
        hbeg, wbeg = np.random.randint(0, th - h), np.random.randint(0, tw - w)
        result_img = img[hbeg:hbeg + h, wbeg:wbeg + w]
        result_mask = mask[hbeg:hbeg + h, wbeg:wbeg + w]
        hcoords[:, 0] -= hcoords[:, 2] * wbeg
        hcoords[:, 1] -= hcoords[:, 2] * hbeg
    else:
        # padding
        result_img = np.zeros([h, w, img.shape[2]], img.dtype)
        result_mask = np.zeros([h, w], mask.dtype)
        hbeg, wbeg = (h - th) // 2, (w - tw) // 2
        result_img[hbeg:hbeg + th, wbeg:wbeg + tw] = img
        result_mask[hbeg:hbeg + th, wbeg:wbeg + tw] = mask
        hcoords[:, 0] += hcoords[:, 2] * wbeg
        hcoords[:, 1] += hcoords[:, 2] * hbeg

    return result_img, result_mask, hcoords


def _augmentation(img, mask, hcoords, height, width):
    foreground = np.sum(mask)

    if foreground > 0:
        # randomly rotate around the center of the instance
        img, mask, hcoords = rotate_instance(img, mask, hcoords, -30, 30)

        # randomly crop and resize
        # 1. firstly crop a region which is [scale_min,scale_max]*[height,width], which ensures that
        #    the area of the intersection between the cropped region and the instance region is at least
        #    overlap_ratio**2 of instance region.
        # 2. if the region is larger than original image, then padding 0
        # 3. then resize the cropped image to [height, width] (bilinear for image, nearest for mask)
        img, mask, hcoords = crop_resize_instance_v1(img, mask, hcoords, height, width, 0.8, 0.8, 1.2)
    else:
        img, mask = crop_or_padding_to_fixed_size(img, mask, height, width)

    return img, mask, hcoords


def augmentation(rgb, mask, hcoords, height, width, split):
    if split == 'train':
        rgb, mask, hcoords = _augmentation(rgb, mask, hcoords, height, width)

        if np.random.random() < 0.5:
            blur_image(rgb, np.random.choice([3, 5, 7, 9]))

        img_transforms = transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        rgb = img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))
    else:
        test_img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        rgb = test_img_transforms(Image.fromarray(np.ascontiguousarray(rgb, np.uint8)))
    return rgb, mask, hcoords

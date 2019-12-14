# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

def calc_2d_bbox(xs, ys, im_size, clip=False):
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    if clip:
        bb_tl = clip_pt_to_im(bb_tl, im_size)
        bb_br = clip_pt_to_im(bb_br, im_size)
    return [bb_tl[0], bb_tl[1], bb_br[0] - bb_tl[0], bb_br[1] - bb_tl[1]]
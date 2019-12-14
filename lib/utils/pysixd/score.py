# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

import numpy as np

def ap(rec, pre):
    '''
    Average Precision (AP) as calculated in the PASCAL VOC challenge from 2010
    onwards [1]:
    1) Compute a version of the measured precision/recall curve with precision
       monotonically decreasing, by setting the precision for recall r to the
       maximum precision obtained for any recall r' >= r.
    2) Compute the AP as the area under this curve by numerical integration.
       No approximation is involved since the curve is piecewise constant.

    NOTE: The used AP formula is different from the one in [2] where the
    formula from VLFeat [3] was presented - although it was mistakenly
    introduced as a formula used in PASCAL.

    References:
    [1] http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00044000000000000000
    [2] Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016
    [3] http://www.vlfeat.org/matlab/vl_pr.html

    :param rec: A list (or 1D ndarray) of recall rates.
    :param pre: A list (or 1D ndarray) of precision rates.
    :return: Average Precision - the area under the monotonically decreasing
             version of the precision/recall curve given by rec and pre.
    '''
    i = np.argsort(rec) # Sorts the precision/recall points by increasing recall
    mrec = np.concatenate(([0], np.array(rec)[i], [1]))
    mpre = np.concatenate(([0], np.array(pre)[i], [0]))
    assert(mrec.shape == mpre.shape)
    for i in range(mpre.size - 3, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap

if __name__ == '__main__':
    # AP test
    tp = np.array([False, True, True, False, True, False])
    fp = np.logical_not(tp)
    tp_c = np.cumsum(tp).astype(np.float)
    fp_c = np.cumsum(fp).astype(np.float)
    rec = tp_c / tp.size
    pre = tp_c / (fp_c + tp_c)
    print('Average Precision: ' + str(ap(rec, pre)))

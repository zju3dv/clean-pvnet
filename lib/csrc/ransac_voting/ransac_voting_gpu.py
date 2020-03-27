import torch
import lib.csrc.ransac_voting.ransac_voting as ransac_voting
import numpy as np


def ransac_voting_layer(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask,:] = cur_win_pts[larger_mask,:]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:,:, 0] = direct[:,:, 1]
        normal[:,:, 1] = -direct[:,:, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier = torch.squeeze(all_inlier.float(), 0)              # [vn,tn]
        normal = normal.permute(1, 0, 2)                                # [vn,tn,2]
        normal = normal*torch.unsqueeze(all_inlier, 2)                 # [vn,tn,2] outlier is all zero

        b = torch.sum(normal*torch.unsqueeze(coords, 0), 2)             # [vn,tn]
        ATA = torch.matmul(normal.permute(0, 2, 1), normal)              # [vn,2,2]
        ATb = torch.sum(normal*torch.unsqueeze(b, 2), 1)                # [vn,2]
        try:
            all_win_pts = torch.matmul(torch.inverse(ATA), torch.unsqueeze(ATb, 2)) # [vn,2,1]
            batch_win_pts.append(all_win_pts[None,:,:, 0])
        except:
            all_win_pts = torch.zeros([1, ATA.size(0), 2]).to(ATA.device)
            batch_win_pts.append(all_win_pts)

    batch_win_pts = torch.cat(batch_win_pts)

    return batch_win_pts

def b_inv(b_mat):
    '''
    code from
    https://stackoverflow.com/questions/46595157/how-to-apply-the-torch-inverse-function-of-pytorch-to-every-sample-in-the-batc
    :param b_mat:
    :return:
    '''
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    try:
        b_inv, _ = torch.solve(eye, b_mat)
    except:
        b_inv = eye
    return b_inv


def ransac_voting_layer_v3(mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
    '''
    :param mask:      [b,h,w]
    :param vertex:    [b,h,w,vn,2]
    :param round_hyp_num:
    :param inlier_thresh:
    :return: [b,vn,2]
    '''
    b, h, w, vn, _ = vertex.shape
    batch_win_pts = []
    for bi in range(b):
        hyp_num = 0
        cur_mask = (mask[bi]).byte()
        foreground_num = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
            batch_win_pts.append(win_pts)  # [1,vn,2]
            continue

        # if too many inliers, we randomly down sample it
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float())).byte()
            cur_mask *= selected_mask

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
        all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn

            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break

        # compute mean intersection again
        normal = torch.zeros_like(direct)   # [tn,vn,2]
        normal[:, :, 0] = direct[:, :, 1]
        normal[:, :, 1] = -direct[:, :, 0]
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
        ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

        # coords [tn,2] normal [vn,tn,2]
        all_inlier = torch.squeeze(all_inlier.float(), 0)              # [vn,tn]
        normal = normal.permute(1, 0, 2)                                # [vn,tn,2]
        normal = normal*torch.unsqueeze(all_inlier, 2)                 # [vn,tn,2] outlier is all zero

        b = torch.sum(normal*torch.unsqueeze(coords, 0), 2)             # [vn,tn]
        ATA = torch.matmul(normal.permute(0, 2, 1), normal)              # [vn,2,2]
        ATb = torch.sum(normal*torch.unsqueeze(b, 2), 1)                # [vn,2]
        # try:
        all_win_pts = torch.matmul(b_inv(ATA), torch.unsqueeze(ATb, 2)) # [vn,2,1]
        # except:
        #    __import__('ipdb').set_trace()
        batch_win_pts.append(all_win_pts[None,:,:, 0])

    batch_win_pts = torch.cat(batch_win_pts)
    return batch_win_pts


def estimate_voting_distribution_with_mean(mask, vertex, mean, round_hyp_num=256, min_hyp_num=4096, topk=128, inlier_thresh=0.99, min_num=5, max_num=30000, output_hyp=False):
    b, h, w, vn, _ = vertex.shape
    all_hyp_pts, all_inlier_ratio = [], []
    for bi in range(b):
        k = 0
        cur_mask = mask[bi] == k + 1
        foreground = torch.sum(cur_mask)

        # if too few points, just skip it
        if foreground < min_num:
            cur_hyp_pts = torch.zeros([1, min_hyp_num, vn, 2], dtype=torch.float32, device=mask.device).float()
            all_hyp_pts.append(cur_hyp_pts)  # [1,vn,2]
            cur_inlier_ratio = torch.ones([1, min_hyp_num, vn], dtype=torch.int64, device=mask.device).float()
            all_inlier_ratio.append(cur_inlier_ratio)
            continue

        # if too many inliers, we randomly down sample it
        if foreground > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground.float()))
            cur_mask *= selected_mask
            foreground = torch.sum(cur_mask)

        coords = torch.nonzero(cur_mask).float()  # [tn,2]
        coords = coords[:, [1, 0]]
        direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
        direct = direct.view([coords.shape[0], vn, 2])
        tn = coords.shape[0]

        round_num = np.ceil(min_hyp_num/round_hyp_num)
        cur_hyp_pts = []
        cur_inlier_ratio = []
        for round_idx in range(int(round_num)):
            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])

            # generate hypothesis
            hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, hyp_pts, inlier, inlier_thresh)  # [hn,vn,tn]
            inlier_ratio = torch.sum(inlier, 2)                     # [hn,vn]
            inlier_ratio = inlier_ratio.float()/foreground.float()    # ratio

            cur_hyp_pts.append(hyp_pts)
            cur_inlier_ratio.append(inlier_ratio)

        cur_hyp_pts = torch.cat(cur_hyp_pts, 0)
        cur_inlier_ratio = torch.cat(cur_inlier_ratio, 0)
        all_hyp_pts.append(torch.unsqueeze(cur_hyp_pts, 0))
        all_inlier_ratio.append(torch.unsqueeze(cur_inlier_ratio, 0))

    all_hyp_pts = torch.cat(all_hyp_pts, 0)               # b,hn,vn,2
    all_inlier_ratio = torch.cat(all_inlier_ratio, 0)     # b,hn,vn

    # raw_hyp_pts=all_hyp_pts.permute(0,2,1,3).clone()
    # raw_hyp_ratio=all_inlier_ratio.permute(0,2,1).clone()

    all_hyp_pts = all_hyp_pts.permute(0, 2, 1, 3)            # b,vn,hn,2
    all_inlier_ratio = all_inlier_ratio.permute(0, 2, 1)    # b,vn,hn
    thresh = torch.max(all_inlier_ratio, 2)[0]-0.1         # b,vn
    all_inlier_ratio[all_inlier_ratio < torch.unsqueeze(thresh, 2)] = 0.0


    diff_pts = all_hyp_pts-torch.unsqueeze(mean, 2)                  # b,vn,hn,2
    weighted_diff_pts = diff_pts * torch.unsqueeze(all_inlier_ratio, 3)
    cov = torch.matmul(diff_pts.transpose(2, 3), weighted_diff_pts)  # b,vn,2,2
    cov /= torch.unsqueeze(torch.unsqueeze(torch.sum(all_inlier_ratio, 2), 2), 3)+1e-3 # b,vn,2,2

    # if output_hyp:
    #     return mean,cov,all_hyp_pts,all_inlier_ratio,raw_hyp_pts,raw_hyp_ratio

    return mean, cov

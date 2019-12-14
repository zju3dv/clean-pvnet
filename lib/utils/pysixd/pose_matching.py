# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

def match_poses(errs, error_thresh, max_ests_count=-1, gt_valid_mask=None):

    # Sort the estimated poses by decreasing confidence score
    errs_s = sorted(errs, key=lambda e: e['score'], reverse=True)

    # If there are more estimated poses than the specified number of instances,
    # keep only the poses with the highest confidence score
    if max_ests_count > 0:
        errs_s = errs_s[:max_ests_count]

    # Greedily match the estimated poses with the ground truth poses
    matches = []
    gt_matched = []
    for e in errs_s:
        best_gt_id = -1
        best_error = float('inf')
        for gt_id, error in e['errors'].items():

            # If the mask of valid GT poses is not provided, consider all valid
            if (not gt_valid_mask or gt_valid_mask[gt_id])\
                     and gt_id not in gt_matched and error < best_error:
                best_gt_id = gt_id
                best_error = error

        if best_error < error_thresh:
            gt_matched.append(best_gt_id)
            best_error_norm = best_error / float(error_thresh)
            matches.append({'est_id': e['est_id'],
                            'gt_id': best_gt_id,
                            'score': e['score'],
                            'error': best_error,
                            'error_norm': best_error_norm})
    return matches

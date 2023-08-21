# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter
import time
import copy
import logging
import numba as nb
from . import tracker
# from multiprocessing import Pool, Process

INFTY_COST = 1e18
# pool = Pool(2)


def get_remain_pair_matrix(cost_matrix, row_indices):
    """
    获取未匹配到的轨迹

    Args:
        cost_matrix ([type]): [description]
        row_indices ([type]): [description]

    Returns:
        [type]: [description]
    """

    track_ids = np.arange(cost_matrix.shape[0])
    row_indices = np.array(row_indices)
    if row_indices.shape[0] > 0:
        remain_track_ids = np.delete(track_ids, row_indices)
        cost_matrix = cost_matrix[remain_track_ids, :]

    return cost_matrix


def better_np_unique(arr):
    """
    获取数组的元素重复信息

    Args:
        arr ([type]): [description]

    Returns:
        [type]: [description]
    """

    sort_indexes = np.argsort(arr)
    arr = np.asarray(arr)[sort_indexes]
    vals, first_indexes, _, counts = np.unique(arr,
                                               return_index=True, return_inverse=True, return_counts=True)
    indexes = np.split(sort_indexes, first_indexes[1:])
    for x in indexes:
        x.sort()
    return vals, indexes, counts


def split_conflict(pre_matches):
    """
    获取冲突与未冲突的匹配（按检测id为准）

    Args:
        pre_matches ([type]): [description]
        detection_indices ([type]): [description]

    Returns:
        [type]: [description]
    """
    vals, indexes, counts = better_np_unique(pre_matches[:, 0])

    # get non conflict matches
    single_idxs = np.where(counts == 1)[0]
    single_det_indices = [int(indexes[i]) for i in single_idxs]
    matches = pre_matches[single_det_indices]

    # get conflict matches
    conflict_matches = []
    multi_idxs = np.where(counts > 1)[0]
    for i in multi_idxs:
        conflict_matches.append(((pre_matches[indexes[i], 0]), vals[i]))

    return list(map(tuple, matches)), conflict_matches


def split_TD_conflict(pre_matches):
    """
    获取冲突与未冲突的匹配（按轨迹id和检测id为准）

    Args:
        pre_matches ([type]): [description]
        detection_indices ([type]): [description]

    Returns:
        [type]: [description]
    """
    T_vals, T_indexes, T_counts = better_np_unique(pre_matches[:, 0])
    D_vals, D_indexes, D_counts = better_np_unique(pre_matches[:, 1])

    matches, matches_ind = [], []
    # get non conflict matches
    # count==1
    single_idxs = np.where(T_counts == 1)[0]
    for i in single_idxs:
        T_ind = int(T_indexes[i])
        dv = pre_matches[T_ind, 1]
        c = D_counts[np.argwhere(D_vals == dv)]
        if c == 1:
            matches.append(pre_matches[T_ind, :2])
            matches_ind.append(T_ind)

    # count==2 & f(indexes[0]) == g(indexes[1])
    two_idxs = np.where(T_counts == 2)[0]
    for i in two_idxs:
        T_ind0 = int(T_indexes[i][0])
        T_ind1 = int(T_indexes[i][1])
        dv0 = pre_matches[T_ind0, 1]
        dv1 = pre_matches[T_ind1, 1]
        if dv0 == dv1:
            matches.append(pre_matches[T_ind0, :2])
            matches_ind.append(T_ind0)
            matches_ind.append(T_ind1)
    matches = np.array(matches, dtype=np.int)
    # matches = matches[matches[:, 0].argsort(), :]

    # get conflict matches
    conflicts_ind = list(
        set(np.arange(pre_matches.shape[0])) - set(matches_ind))
    conflict_matches = pre_matches[conflicts_ind, :]

    return list(map(tuple, matches)), conflict_matches


def _assign(level, cost_matrix, track_info, track_indices, detection_indices, gate):
    cost_matrix = filter_pairs(cost_matrix, gate)
    row_indices, col_indices = linear_assignment(cost_matrix)

    pre_matches = []
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] <= gate:
            pre_matches.append(
                (track_idx, detection_idx, track_info[track_idx]))
    pre_matches = np.array(pre_matches)

    if pre_matches.shape[0]:
        unmatched_tracks = list(set(track_indices) -
                                set(list(pre_matches[:, 0])))
        unmatched_detections = list(set(detection_indices) -
                                    set(list(pre_matches[:, 1])))
        if level == 0:
            pre_matches = np.hstack(
                (pre_matches, np.zeros(pre_matches.shape[0]).reshape(-1, 1)))
        else:
            pre_matches = np.hstack(
                (pre_matches, np.ones(pre_matches.shape[0]).reshape(-1, 1)))
    else:
        unmatched_tracks = track_indices
        unmatched_detections = detection_indices

    return pre_matches, unmatched_tracks, unmatched_detections



def dist_cost_matrix(track_previous_ct, det_previous_ct, tracks, dets, motion_gate=1.1e18, overlap_thresh=0.05):
    cost_matrix = (((det_previous_ct.reshape(1, -1, 2) -
                   track_previous_ct.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # M x N
    # motion threshold
    tracks_wh = np.array([track.tlwh[2:4] for track in tracks])
    track_size = tracks_wh[:, 0] * tracks_wh[:, 1]
    det_wh = np.array([det.tlwh[2:4] for det in dets])
    item_size = det_wh[:, 0] * det_wh[:, 1]

    # iou
    track_boxes = np.array([[track.tlwh[0],
                           track.tlwh[1],
                           track.tlwh[0]+track.tlwh[2],
                           track.tlwh[1]+track.tlwh[3]]
                            for track in tracks], np.float32)  # M x 4
    det_boxes = np.array([[det.tlwh[0],
                         det.tlwh[1],
                         det.tlwh[0]+det.tlwh[2],
                         det.tlwh[1]+det.tlwh[3]]
                          for det in dets], np.float32)  # N x 4
    box_ious = bbox_overlaps_py(track_boxes, det_boxes)

    # gate filter
    invalid = ((cost_matrix > track_size.reshape(-1, 1))
               + (cost_matrix > item_size.reshape(1, -1))
               + (box_ious < overlap_thresh)
               ) > 0
    cost_matrix = cost_matrix + invalid * motion_gate

    return cost_matrix


def bbox_overlaps_py(boxes, query_boxes):
    """
    determine overlaps between boxes and query_boxes
    :param boxes: n * 4 bounding boxes
    :param query_boxes: k * 4 bounding boxes
    :return: overlaps: n * k overlaps
    """
    n_ = boxes.shape[0]
    k_ = query_boxes.shape[0]
    overlaps = np.zeros((n_, k_), dtype=np.float)
    for k in range(k_):
        query_box_area = (
            query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(n_):
            iw = min(boxes[n, 2], query_boxes[k, 2]) - \
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) - \
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    box_area = (boxes[n, 2] - boxes[n, 0] + 1) * \
                        (boxes[n, 3] - boxes[n, 1] + 1)
                    all_area = float(box_area + query_box_area - iw * ih)
                    overlaps[n, k] = iw * ih / all_area
    return overlaps


def cost_metric(metric, sigma, level, tracks, dets, track_indices, detection_indices):
    features = np.array([dets[i].feature for i in detection_indices])
    targets = np.array([tracks[i].track_id for i in track_indices])
    tracks = [tracks[i] for i in track_indices]
    dets_use = [dets[i] for i in detection_indices]

    det_previous_ct = np.array([d.previous_ct for d in dets_use])
    track_previous_ct = np.array([t.ct for t in tracks])

    # motion level
    delta_xy = np.array([t.mean[4:6] for t in tracks])
    alpha = tracker.metric_gaussian_motion(delta_xy, sigma)
    if level == 0:
        start_time = time.time()
        # motion cost matrix
        motion_cost = dist_cost_matrix(
            track_previous_ct, det_previous_ct, tracks, dets_use)
        motion_time = time.time()
        motion_t = motion_time - start_time
        # print("motion_t: %.4f" % motion_t)
        cost_matrix = motion_cost
    else:
        # appearance cost matrix
        start_time = time.time()
        appearance_cost = metric.distance(features, targets)
        appear_time = time.time()
        appear_t = appear_time - start_time
        # print("appear_t: %.4f" % appear_t)
        cost_matrix = appearance_cost

    tracks_info = {}
    for i, idx in enumerate(track_indices):
        tracks_info[idx] = alpha[i]

    return cost_matrix, tracks_info


def metric_assign(metric, sigma, level, tracks, detections, track_indices, detection_indices, metric_gate):
    # metric assign
    start_time = time.time()
    # cost_matrix, info = distance_metric(
    #     sigma, level, tracks, detections, track_indices, detection_indices)
    cost_matrix, info = cost_metric(metric,
                                    sigma, level, tracks, detections, track_indices, detection_indices)

    dis_time = time.time()
    dis_t = dis_time - start_time

    # 0.0002 sec
    matches, unmatched_tracks, unmatched_detections = _assign(
        level, cost_matrix, info, track_indices, detection_indices, metric_gate)
    assign_time = time.time()
    assign_t = assign_time - dis_time
    # print("dis_t: %.4f | assign_t: %.4f" % (dis_t, assign_t), end="|")
    return (matches, unmatched_tracks, unmatched_detections)

    # return cost_matrix, info


def test(a):
    time.sleep(0.002)
    a = np.array([(1, 1, 1, 1)] * 15)
    return a, a, a


def pre_assignment(opt, level, tracks, detections, track_indices, detection_indices, metric, motion_distance, appearance_distance):
    """
    为每个track分配一个det

    Args:
        track_indices ([type]): [description]
        detection_indices ([type]): [description]

    Returns:
        [type]: [description]
    """

    if level == 0:
        # """start-----------single process"""
        start_time = time.time()
        results = []
        for i, gate in enumerate([motion_distance, appearance_distance]):
            results.append(
                metric_assign(metric, opt.sigma, i, tracks, detections, track_indices, detection_indices, gate))

        results = [r for r in results]

        single_time = time.time()
        single_t = single_time - start_time
        # """single process-------------end"""

        """start-----------multi process
        pool version
        """
        # gates = [motion_distance, appearance_distance]
        # res = []
        # for i in range(2):
        #     # res.append(
        #     #     opt.pool.apply_async(func=test,
        #     #                          args=(1,)))
        #     # for i, gate in enumerate([motion_distance, appearance_distance]):
        #     res.append(
        #         opt.pool.apply_async(func=metric_assign,
        #                              args=(metric, opt.sigma, i, tracks, detections, track_indices, detection_indices, gates[i],)))
        # results = [r.get() for r in res]

        # mutli_time = time.time()
        # mutli_t = mutli_time - single_time
        # print("single_t: %.4f | mutli_t: %.4f" %
        #       (single_t, mutli_t))
        # """multi process--------------end"""

        motion_results = results[0]
        appearance_results = results[1]

        pre_matches = np.vstack((motion_results[0], appearance_results[0])
                                ) if appearance_results[0].shape[0] else motion_results[0]
        unmatched_tracks = list(set(motion_results[1]).intersection(
            set(appearance_results[1])))
        unmatched_detections = list(set(motion_results[2]).intersection(
            set(appearance_results[2])))

        return pre_matches, unmatched_tracks, unmatched_detections
    else:
        # second metric assign
        appearance_matches, appearance_unmatched_tracks, appearance_unmatched_detections = metric_assign(
            metric, opt.sigma, 1, tracks, detections, track_indices, detection_indices, appearance_distance)

        return appearance_matches, list(appearance_unmatched_tracks), list(appearance_unmatched_detections)


def pre_assignmentv1(cost_matrix, tracks, detections, track_indices, detection_indices, gate, max_distance, distance_metric):
    """
    为每个track分配一个det

    Args:
        track_indices ([type]): [description]
        detection_indices ([type]): [description]

    Returns:
        [type]: [description]
    """
    unmatched_track_indices = copy.deepcopy(track_indices)
    pre_matches, unmatched_tracks, unmatched_detections, = [], [], []
    # every tracklets should be matched one det bbox
    while len(unmatched_track_indices):
        row_indices, col_indices = linear_assignment(cost_matrix)

        cur_assignment_idx = []
        for row, col in zip(row_indices, col_indices):
            track_idx = unmatched_track_indices[row]
            detection_idx = detection_indices[col]

            if cost_matrix[row, col] > gate:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                pre_matches.append((track_idx, detection_idx))

            # pre_matches.append((track_idx, detection_idx))
            cur_assignment_idx.append(track_idx)

        for idx in cur_assignment_idx:
            unmatched_track_indices.remove(idx)

        # get remained pair matrix
        cost_matrix = get_remain_pair_matrix(cost_matrix, row_indices)

        # appearance distance for remained track
        if len(unmatched_track_indices):
            gate = max_distance

            # distance metric
            cost_matrix, _ = distance_metric(
                1, tracks, detections, unmatched_tracks, unmatched_detections)
            cost_matrix = filter_pairs(cost_matrix, gate=gate)

    pre_matches = np.array(sorted(pre_matches, key=lambda p: p[0]))

    return pre_matches, unmatched_tracks


def extract_conflict_tracks_info(tracklet_idx, det_id_index, tracks_info, alpha_thredhold):
    fisrt_tracks, second_tracks = [], []
    for idx in tracklet_idx:
        appearance_cost = tracks_info[idx]["appearance_cost"][det_id_index]
        # motion_cost = tracks_info[idx]["motion_cost"][det_id_index]
        miss_nums = tracks_info[idx]["miss_nums"]
        if tracks_info[idx]['alpha'] >= alpha_thredhold:
            fisrt_tracks.append((idx, appearance_cost,  miss_nums))
            # fisrt_tracks.append((idx, appearance_cost, motion_cost, miss_nums))
        else:
            second_tracks.append(
                (idx, appearance_cost, miss_nums))
            # (idx, appearance_cost, motion_cost, miss_nums))
    fisrt_tracks = np.array(fisrt_tracks)
    second_tracks = np.array(second_tracks)

    return fisrt_tracks, second_tracks


def first_round_association(fisrt_tracks, second_tracks, det_id, tolerate_threshold):
    """
    first round association
    Motion and high similarity trajectories are assigned first
    """
    if len(fisrt_tracks) == 0:
        return [], []

    # compare appearance distance
    if second_tracks.shape[0] > 0:
        min_second_appearance = np.min(second_tracks[:, 1])
        accepted_appearance = np.where((fisrt_tracks[:, 1] - min_second_appearance) /
                                       min_second_appearance <= tolerate_threshold, True, False)
    else:
        accepted_appearance = np.array(fisrt_tracks.shape[0] * [True])

    matches, unmatched_tracks = [], []
    # matches
    matches_tracks_ids = fisrt_tracks[accepted_appearance, 0]
    matches = [(int(i), det_id) for i in matches_tracks_ids]
    # unmatched
    unmatched_tracks = list(fisrt_tracks[~accepted_appearance, 0])

    if len(matches):
        # logging.info(f"matches: {matches}")
        return [matches[0]], unmatched_tracks
    else:
        return matches, unmatched_tracks


def second_round_association(second_tracks, det_id, max_discount_num, det_used):
    """
    second round association other trajectories are assigned by their own predicted bbox
    """
    matches, unmatched_tracks, kf_ids = [], [], []
    while second_tracks.shape[0]:
        if not det_used:
            # get the track id of min appearance cost
            min_id = np.argmin(second_tracks[:, 1])
            track_id = second_tracks[min_id, 0]
            matches.append((int(track_id), det_id))
            # remove id
            remain_ids = [True] * second_tracks.shape[0]
            remain_ids[min_id] = False
            second_tracks = second_tracks[remain_ids, :]
        else:
            discount = np.where(
                second_tracks[:, -1] <= max_discount_num, True, False)
            # TODO:应该要考虑匹配阈值
            kf_ids = list(second_tracks[discount, 0].astype(np.int32))
            unmatched_tracks = list(second_tracks[~discount, 0])

            break

    return matches, unmatched_tracks, kf_ids


def two_round_match_v1(conflict_matches, tracks_info, detection_indices, alpha_thredhold=0.5, tolerate_threshold=0.1, max_discount_num=10):
    # TODO: 这里有3个超参数
    matches, unmatched_tracks, kf_ids = [], [], []
    for pair in conflict_matches:
        tracklet_idx, det_id = pair[0], pair[1]

        # extract conflict tracks info for tow-round match
        det_id_index = detection_indices.index(det_id)
        fisrt_tracks, second_tracks = extract_conflict_tracks_info(
            tracklet_idx, det_id_index, tracks_info, alpha_thredhold)

        # first round association
        matches_first, unmatched_tracks_first = first_round_association(
            fisrt_tracks, second_tracks, det_id, tolerate_threshold)

        # second round association
        det_used = False if len(matches_first) == 0 else True
        matches_second, unmatched_tracks_second, kf_ids = second_round_association(
            second_tracks, det_id, max_discount_num, det_used)

        # merge two-round match
        matches = matches_first + matches_second
        unmatched_tracks = unmatched_tracks_first + unmatched_tracks_second

    return matches, unmatched_tracks, kf_ids


def filter_pairs(cost_matrix, gate):
    cost_matrix[cost_matrix > gate] = gate + INFTY_COST

    return cost_matrix


def iou_cost_matching(
    distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    # distance metric
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices)

    cost_matrix[cost_matrix > max_distance] = max_distance + INFTY_COST

    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


def select_pairs(matches, col_id, pairs):
    for t in pairs:
        ind = np.argwhere(matches[:, col_id] == t)
        if len(ind):
            matches[ind[0], -1] = 0
    matches = matches[matches[:, -1] == 1, :]

    return matches


def two_round_match(conflicts, alpha_gate=0.9):
    if conflicts.shape[0] == 0:
        return [], [], []
    
    # appearance match
    first_round = conflicts[conflicts[:, -2] >= alpha_gate, :]
    matches_a = first_round[first_round[:, -1] == 1, :2]

    # motion match
    second_round = conflicts[conflicts[:, -2] < alpha_gate, :]
    second_round = second_round[second_round[:, -1] == 0, :]
    mask = np.ones(second_round.shape[0]).reshape(-1, 1)
    second_round = np.hstack((second_round, mask))

    # filter track
    second_round = select_pairs(second_round, 0, matches_a[:, 0])

    # filter det
    second_round = select_pairs(second_round, 1, matches_a[:, 1])
    matches_b = second_round[:, :2]

    # match results
    matches_ = np.vstack((matches_a, matches_b))
    matches_ = matches_.astype(np.int)
    unmatched_tracks_ = list(set(conflicts[:, 0]) - set(matches_[:, 0]))
    unmatched_tracks_ = [int(i) for i in unmatched_tracks_]
    unmatched_detections_ = list(set(conflicts[:, 1]) - set(matches_[:, 1]))
    unmatched_detections_ = [int(i) for i in unmatched_detections_]

    return list(map(tuple, matches_)), unmatched_tracks_, unmatched_detections_


def min_cost_matching(
        opt, level, metric, appearance_distance, tracks, detections, track_indices=None,
        detection_indices=None, motion_distance=1e+18):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices, []  # Nothing to match.

    # """[start]--------------------deepsort--------------------"""
    # # distance metric
    # cost_matrix, tracks_info = distance_metric(level,
    #                                            tracks, detections, track_indices, detection_indices)
    # gate = motion_distance if level == 0 else appearance_distance
    # cost_matrix = filter_pairs(cost_matrix, gate=gate)
    # row_indices, col_indices = linear_assignment(cost_matrix)
    # matches, unmatched_tracks, unmatched_detections = [], [], []
    # for col, detection_idx in enumerate(detection_indices):
    #     if col not in col_indices:
    #         unmatched_detections.append(detection_idx)
    # for row, track_idx in enumerate(track_indices):
    #     if row not in row_indices:
    #         unmatched_tracks.append(track_idx)
    # for row, col in zip(row_indices, col_indices):
    #     track_idx = track_indices[row]
    #     detection_idx = detection_indices[col]
    #     if cost_matrix[row, col] > motion_distance:
    #         unmatched_tracks.append(track_idx)
    #         unmatched_detections.append(detection_idx)
    #     else:
    #         matches.append((track_idx, detection_idx))

    # return matches, unmatched_tracks, unmatched_detections, []
    # """--------------------deepsort--------------------[end]"""

    """[start]--------------------two-round--------------------"""
    start_time = time.time()
    pre_matches, unmatched_tracks, unmatched_detections = pre_assignment(opt, level, tracks, detections,
                                                                         track_indices, detection_indices,
                                                                         metric, motion_distance, appearance_distance)
    preassign_time = time.time()
    preassign_t = preassign_time - start_time

    if pre_matches.shape[0]:
        matches, conflicts = split_TD_conflict(pre_matches)
        conflict_time = time.time()
        conflict_t = conflict_time - preassign_time

        matches_, unmatched_tracks_, unmatched_detections_ = two_round_match(
            conflicts, opt.alpha_gate)
        tworound_time = time.time()
        tworound_t = tworound_time - conflict_time
        tot_t = time.time() - start_time
        matches += matches_
        unmatched_tracks += unmatched_tracks_
        unmatched_detections += unmatched_detections_
        kf_ids = []
        # print("tot_t: %.4f | preassign_t: %.4f | conflict_t: %.4f | tworound_t: %.4f" %
        #       (tot_t, preassign_t, conflict_t, tworound_t))
    else:
        matches = []
        kf_ids = []

    matches = sorted(matches, key=lambda m: m[0])
    return matches, unmatched_tracks, unmatched_detections, kf_ids
    """--------------------two-round--------------------[end]"""


def matching_cascade(opt,
                     metric, max_distance, cascade_depth, tracks, detections,
                     track_indices=None, detection_indices=None):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches, kf_ids = [], []
    # TODO:除了level=0，其他的级联匹配的其他level只能用appearance做
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break
        # 获取当前level的trk 索引号
        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections, kf_ids_l = min_cost_matching(opt,
                                                                         level, metric, max_distance, tracks, detections,
                                                                         track_indices_l, unmatched_detections)

        matches += matches_l
        kf_ids += kf_ids_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

    return matches, unmatched_tracks, unmatched_detections, kf_ids


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix


def motion_cost_matrix(
        kf, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns motion cost matrix.

    """
    def normalization(data):
        # TODO：思考更合理的归一化方法
        if data.shape[0] == 1:
            data[0, 0] = 1.0
            return data
        else:
            _range = np.max(data) - np.min(data)
            return (data - np.min(data)) / _range

    # gating_dim = 2 if only_position else 4
    # gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices])

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        track = tracks[row]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)

        cost_matrix[row, :] = gating_distance
        # TODO：是否需要加这个过滤？应该是不需要！
        # gating_distance[row, gating_distance > gating_threshold] = gated_cost

    return normalization(cost_matrix)

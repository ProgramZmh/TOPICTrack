from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
import torch.nn.functional as F
import sys
INFTY_COST = 999


def metric_gaussian_motion(alpha, sigma):
    """以高斯核的方式计算轨迹的运动水平"""
    # use time: 0.0001 ~ 0.0005

    # alpha = 1. - np.exp(- (delta_xy / (2 * sigma**2)))
    # alpha = 1. - np.exp(- (delta_xy[:, 0] ** 2 +
    #                        delta_xy[:, 1]**2) / (2 * sigma**2))

    return alpha.astype(np.float32)


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) *
        (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) *
        (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )
    return o

def softmax_by_row(arr):
    exp_arr = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    softmax_result = exp_arr / np.sum(exp_arr, axis=1, keepdims=True)
    return softmax_result

def calculate_aspect_ratios(det, trk):
    def aspect_ratio(bbox):
        width = bbox[:, 2] - bbox[:, 0]
        height = bbox[:, 3] - bbox[:, 1]
        return width / height

    det_aspect_ratios = aspect_ratio(det)
    trk_aspect_ratios = aspect_ratio(trk)

    # Calculate the absolute difference in aspect ratios between each pair of bbox
    abs_diff_aspect_ratios = np.abs(trk_aspect_ratios[:, np.newaxis] - det_aspect_ratios)

    # abs_diff_aspect_ratios = softmax_by_row(abs_diff_aspect_ratios)
    return abs_diff_aspect_ratios

def set_non_min_to_one_by_row(arr,val=1):
    # Find the minimum value in each row
    min_values = np.min(arr, axis=1, keepdims=True)
    
    # Create a mask for non-minimum values in each row
    mask = arr > min_values
    
    # Set non-minimum values to 1
    arr[mask] = val

    return arr
def appearance_associate(dets_embs,
                         trk_embs,
                         dets, tracks,
                         track_indices, detection_indices, tracks_info,
                         gate, iou_threshold,metric,rotate=False):
    # targets = np.array([t.id for t in tracks if t.hits >= min_hits])
    if (len(dets_embs) == 0) or len(trk_embs) == 0:
        return (np.empty((0, 2), dtype=int), np.empty((0, 5), dtype=int), np.arange(len(dets_embs)))
    # print('det_embs: ',dets_embs, 'trk_embs: ',trk_embs)
    cost_matrix, sim_matrix = cal_cost_matrix(dets_embs, trk_embs, metric)
    
    iou_matrix = iou_batch(dets, tracks)

    # 用运动过滤外观
    iou_matrix = filter_pairs(iou_matrix, gate)
    iou_matrix[iou_matrix<INFTY_COST] = 1
    iou_matrix[iou_matrix==INFTY_COST] = 0 # 过滤
    
    iou_matrix = iou_matrix.T
    # emb_cost = emb_cost.T
    cost_matrix = cost_matrix * iou_matrix


    # # 法1：直接加权emb矩阵，再过滤
    # # 宽高比
    # abs_diff_aspect_ratios = calculate_aspect_ratios(dets, tracks)
    # w = 0.8
    # cost_matrix  = w * cost_matrix + (1-w) * abs_diff_aspect_ratios
    

    # # 法2：填补小于长宽比阈值 & emb中INF的值
    # # 宽高比
    if rotate:
        abs_diff_aspect_ratios = calculate_aspect_ratios(dets, tracks)

        abs_diff_aspect_ratios = set_non_min_to_one_by_row(abs_diff_aspect_ratios)
        abs_diff_aspect_ratios[abs_diff_aspect_ratios>0.03] = 1
        abs_diff_aspect_ratios[abs_diff_aspect_ratios<=0.03] = gate-0.0001
    
    # 过滤
    # 运动过滤外观注释以下代码
    # cost_matrix = filter_pairs(cost_matrix, gate)
    
    if rotate:
        cost_matrix[cost_matrix==INFTY_COST] = abs_diff_aspect_ratios[cost_matrix==INFTY_COST]

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # iou_matrix = iou_batch(tracks, dets)
    pre_matches = []
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        # if iou_matrix[row, col] < iou_threshold:  # 运动过快的不要
        #     continue
        if cost_matrix[row, col] <= gate:
            pre_matches.append(
                (track_idx, detection_idx, tracks_info[track_idx]))
    # pre_matches = []  # 相当于禁用外观
    pre_matches = np.array(pre_matches)

    if pre_matches.shape[0]:
        unmatched_tracks = list(set(track_indices) -
                                set(list(pre_matches[:, 0])))
        unmatched_detections = list(set(detection_indices) -
                                    set(list(pre_matches[:, 1])))

        pre_matches = np.hstack(
            (pre_matches, np.ones(pre_matches.shape[0]).reshape(-1, 1)))
    else:
        unmatched_tracks = track_indices
        unmatched_detections = detection_indices
    # print('pre_mathes: ', pre_matches)
    return (pre_matches, unmatched_tracks, unmatched_detections),  abs(sim_matrix.T)


def min_cost_matching(
        motion_pre_assign, appearance_pre_assign,
        alpha_gate=0.9, two_round_off=None):

    # if True:
    #     return np.array(motion_pre_assign[0]), np.array(motion_pre_assign[1]), np.array(motion_pre_assign[2])
    # 预匹配处理
    pre_matches, unmatched_tracks, unmatched_detections = \
        pre_match_process(motion_pre_assign, appearance_pre_assign)
    if pre_matches.shape[0]:
        # # 保存
        # import os
        # file_path = 'speed_sigma@0.5.txt'
        # new_data = np.array(pre_matches[:, 2:3])
        # if os.path.isfile(file_path):
        #     data = np.loadtxt(file_path)
        #     if data.ndim == 1:
        #         data = data.reshape(-1, 1)
        #     new_data = np.array([new_data.max(), new_data.min(),
        #                         new_data.mean()]).reshape(-1, 1)
        #     new_data = np.concatenate((data, new_data), axis=1)
        # else:
        #     new_data = np.array(pre_matches[:, 2:3])
        #     new_data = np.array([new_data.max(), new_data.min(),
        #                         new_data.mean()]).reshape(-1, 1)
        # np.savetxt(file_path, new_data)
        matches, conflicts = split_TD_conflict(pre_matches)
        if len(conflicts):
            print("appr: ", len(appearance_pre_assign[0]),end="\t")
            print("motion: ", len(motion_pre_assign[0]),end="\t")
            print("conflicts: ", len(conflicts),end="\t")
        
        matches_, unmatched_tracks_, unmatched_detections_ = two_round_match(
            conflicts, alpha_gate)

        matches += matches_
        unmatched_tracks += unmatched_tracks_
        unmatched_detections += unmatched_detections_
    else:
        matches = np.empty((0, 2), dtype=int)

    matches = sorted(matches, key=lambda m: m[0])

    return np.array(matches), np.array(unmatched_tracks), np.array(unmatched_detections)


def cal_cost_matrix(dets_embs, trk_embs, metric):
    # features = np.array([d.feature for d in dets])
    # targets = np.array([t.id for t in trackers if t.hits >= min_hits])
    # cost_matrix = metric.distance(features, targets)
    print('metric：',metric)
    if metric == 'res_recons':
    # 法1：外观重建，用外观更新
        cost_matrix, sim_matrix = _nn_res_recons_cosine_distance(
            trk_embs, dets_embs, data_is_normalized=False)
    else:
        # 法2：余弦
        cost_matrix, sim_matrix = _cosine_distance(trk_embs, dets_embs)
        print("cosine cost_matrix.min:%.2f, max:%.2f, mean:%.2f, std:%.2f" %
            (cost_matrix.min(), cost_matrix.max(), cost_matrix.mean(), cost_matrix.std()))
        
    # print("sim_matrix[max=%.2f, min=%.2f]" % (sim_matrix.max(),sim_matrix.min()))
    # 法1.1：外观重建，有外观描述库
    # cost_matrix = np.zeros((len(trk_embs), dets_embs.shape[0]))
    # for i in range(len(trk_embs)):
    #     t_emb = trk_embs[i]
    #     distances = _nn_res_recons_cosine_distance(
    #         t_emb, dets_embs, data_is_normalized=True)
    #     # distances = _cosine_distance(t_emb, dets_embs, data_is_normalized=True)
    #     distances = np.nan_to_num(distances, nan=9999)
    #     distances = distances.min(axis=0)
    #     cost_matrix[i, :] = distances

    # 按轨迹归一化
    # cost_matrix /= np.linalg.norm(cost_matrix, axis=1, keepdims=True)

    # dets_embs = dets_embs.reshape(dets_embs.shape[0], dets_embs.shape[1], 1)
    # cost_matrix = np.zeros((len(trk_embs), dets_embs.shape[0]))
    # for i in range(len(trk_embs)):
    #     t_emb = np.array(trk_embs[i:i+1])
    #     t_emb = np.transpose(t_emb, (0, 2, 1))

    #     distances = reconsdot_distance(t_emb, dets_embs)
    #     distances = np.nan_to_num(distances, nan=9999)
    #     distances = distances.min(axis=0)
    #     cost_matrix[i, :] = distances

    # 法2：余弦
    # cost_matrix, sim_matrix = _cosine_distance(trk_embs, dets_embs)
    # print("cosine cost_matrix.min:%.2f, max:%.2f, mean:%.2f, std:%.2f" %
    #       (cost_matrix.min(), cost_matrix.max(), cost_matrix.mean(), cost_matrix.std()))
    # print('cost_matrix: ', cost_matrix)
    return cost_matrix, sim_matrix


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    sim = np.dot(a, b.T)
    return 1.-sim, sim


def reconsdot_distance(x, y, tmp=100):
    x = torch.from_numpy(np.asarray(x)).half().cuda()
    y = torch.from_numpy(np.asarray(y)).cuda()
    # x = torch.from_numpy(np.asarray(x)).cuda()
    # y = torch.from_numpy(np.asarray(y)).cuda()

    track_features = F.normalize(x, dim=1)
    det_features = F.normalize(y, dim=1)

    ndet, ndim, nsd = det_features.shape
    ntrk, _, nst = track_features.shape

    fdet = det_features.permute(0, 2, 1).reshape(-1, ndim).cuda()
    ftrk = track_features.permute(0, 2, 1).reshape(-1, ndim).cuda()

    aff = torch.mm(ftrk, fdet.transpose(0, 1))
    aff_td = F.softmax(tmp*aff, dim=1)
    aff_dt = F.softmax(tmp*aff, dim=0).transpose(0, 1)

    recons_ftrk = torch.einsum('tds,dsm->tdm', aff_td.view(ntrk*nst, ndet, nsd),
                               fdet.view(ndet, nsd, ndim))
    recons_fdet = torch.einsum('dts,tsm->dtm', aff_dt.view(ndet*nsd, ntrk, nst),
                               ftrk.view(ntrk, nst, ndim))

    recons_ftrk = recons_ftrk.permute(0, 2, 1).reshape(ntrk, nst*ndim, ndet)
    # recons_ftrk = recons_ftrk.permute(0, 2, 1).view(ntrk, nst*ndim, ndet)
    recons_ftrk_norm = F.normalize(recons_ftrk, dim=1)
    recons_fdet = recons_fdet.permute(0, 2, 1).reshape(ndet, nsd*ndim, ntrk)
    # recons_fdet = recons_fdet.permute(0, 2, 1).view(ndet, nsd*ndim, ntrk)
    recons_fdet_norm = F.normalize(recons_fdet, dim=1)

    dot_td = torch.einsum('tad,ta->td', recons_ftrk_norm,
                          F.normalize(ftrk.reshape(ntrk, nst*ndim), dim=1))
    dot_dt = torch.einsum('dat,da->dt', recons_fdet_norm,
                          F.normalize(fdet.reshape(ndet, nsd*ndim), dim=1))

    cost_matrix = 1 - 0.5 * (dot_td + dot_dt.transpose(0, 1))
    cost_matrix = cost_matrix.detach().cpu().numpy()

    return cost_matrix


def res_recons_cosine_distance(x, y, tmp=1):
    x = torch.from_numpy(np.asarray(x)).cuda()
    y = torch.from_numpy(np.asarray(y)).cuda()

    track_features = F.normalize(x, dim=1)
    det_features = F.normalize(y, dim=1)

    ndet, ndim, nsd = det_features.shape
    ntrk, _, nst = track_features.shape

    fdet = det_features.permute(0, 2, 1).reshape(-1, ndim).cuda()
    ftrk = track_features.permute(0, 2, 1).reshape(-1, ndim).cuda()

    aff = torch.mm(ftrk, fdet.transpose(0, 1))
    aff_td = F.softmax(tmp*aff, dim=1)
    aff_dt = F.softmax(tmp*aff, dim=0).transpose(0, 1)

    recons_ftrk = torch.einsum('tds,dsm->tdm', aff_td.view(ntrk*nst, ndet, nsd),
                               fdet.view(ndet, nsd, ndim))
    recons_fdet = torch.einsum('dts,tsm->dtm', aff_dt.view(ndet*nsd, ntrk, nst),
                               ftrk.view(ntrk, nst, ndim))

    recons_ftrk = ftrk + recons_ftrk
    recons_fdet = fdet + recons_fdet
    recons_ftrk_norm = F.normalize(recons_ftrk, dim=1)
    recons_fdet_norm = F.normalize(recons_fdet, dim=1)
    distances = 1 - torch.mm(recons_ftrk_norm,
                             recons_fdet_norm.transpose(0, 1))
    # ==================================================================================

    distances = distances.detach().cpu().numpy()

    return distances
    # return distances.min(axis=0)


def _nn_res_recons_cosine_distance(x, y, tmp=100, data_is_normalized=False):
    if not data_is_normalized:
        x = np.asarray(x) / np.linalg.norm(x, axis=1, keepdims=True)
        y = np.asarray(y) / np.linalg.norm(y, axis=1, keepdims=True)
        # ftrk = F.softmax(ftrk, dim=1)
        # fdet = F.softmax(fdet, dim=1)
        # ftrk = F.normalize(ftrk, dim=1)
        # fdet = F.normalize(fdet, dim=1)
        
    ftrk = torch.from_numpy(np.asarray(x)).half().cuda()  # (5,128)
    fdet = torch.from_numpy(np.asarray(y)).half().cuda()  # (10,128)
    aff = torch.mm(ftrk, fdet.transpose(0, 1))  # (5,10)
    aff_td = F.softmax(tmp*aff, dim=1)
    aff_dt = F.softmax(tmp*aff, dim=0).transpose(0, 1)

    # 重构外观特征
    res_recons_ftrk = torch.mm(aff_td, fdet)  # (5,128)
    res_recons_fdet = torch.mm(aff_dt, ftrk)  # (10,128)
    # res_recons_ftrk = F.normalize(res_recons_ftrk, dim=1)
    # res_recons_fdet = F.normalize(res_recons_fdet, dim=1)

    # 残差 t x (c+1)
    # 方案1：
    # recons_ftrk = ftrk + res_recons_ftrk
    # recons_fdet = fdet + res_recons_fdet
    # recons_ftrk_norm = F.normalize(recons_ftrk, dim=1)
    # recons_fdet_norm = F.normalize(recons_fdet, dim=1)
    # # 计算余弦距离
    # distances = 1 - torch.mm(recons_ftrk_norm,
    #                          recons_fdet_norm.transpose(0, 1))

    # 方案2：
    # s1 =torch.mm(ftrk, fdet.transpose(0, 1))
    # s2 = torch.mm(res_recons_ftrk,res_recons_fdet.transpose(0, 1))
    
    # print("ftrk:[max:%.2f,min:%.2f]" %(ftrk.max(),ftrk.min()),end="\t")
    # print("fdet:[max:%.2f,min:%.2f]" %(fdet.max(),fdet.min()) ,end="\t")
    # print("res_recons_ftrk:[max:%.2f,min:%.2f]" %(res_recons_ftrk.max(),res_recons_ftrk.min()) ,end="\t")
    # print("res_recons_fdet:[max:%.2f,min:%.2f]" %(res_recons_fdet.max(),res_recons_fdet.min()) ,end="\t")
    # print("s1:[max:%.2f,min:%.2f]" %(s1.max(),s1.min()) ,end="\t")
    # print("s2:[max:%.2f,min:%.2f]" %(s2.max(),s2.min()) ,end="\t")
    
    sim = (torch.mm(ftrk, fdet.transpose(0, 1)) + torch.mm(res_recons_ftrk,
                                                           res_recons_fdet.transpose(0, 1))) / 2
    distances = 1-sim
    # print("distances:[max:%.2f,min:%.2f]" %(distances.max(),distances.min()) ,end="\t")
    # distances_1 = 1 - torch.mm(ftrk, fdet.transpose(0, 1))
    # distances_2 = 1 - torch.mm(res_recons_ftrk,
    #                            res_recons_fdet.transpose(0, 1))
    # distances = 0.5 * (distances_1 + distances_2)

    # 方案3：
    # distances_1 = torch.mm(ftrk, fdet.transpose(0, 1))
    # distances_2 = torch.mm(res_recons_ftrk, res_recons_fdet.transpose(0, 1))

    # distances = 1 - 0.5 * (distances_1 + distances_2)

    distances = distances.detach().cpu().numpy()
    sim = sim.detach().cpu().numpy()
    # print('distances: ', distances)
    return distances, sim


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
            # 在这里获取前面列和最一列（我们增加的（0/1））

            # matches.append(pre_matches[T_ind, :2])
            matches.append(pre_matches[T_ind, :])

            matches_ind.append(T_ind)

    # count==2 & f(indexes[0]) == g(indexes[1])
    two_idxs = np.where(T_counts == 2)[0]
    for i in two_idxs:
        T_ind0 = int(T_indexes[i][0])
        T_ind1 = int(T_indexes[i][1])
        dv0 = pre_matches[T_ind0, 1]
        dv1 = pre_matches[T_ind1, 1]
        if dv0 == dv1:
            # 在这里获取前面列和最一列（我们增加的（0/1））

            # matches.append(pre_matches[T_ind0, :2])
            matches.append(pre_matches[T_ind0, :])

            matches_ind.append(T_ind0)
            matches_ind.append(T_ind1)
    # matches = np.array(matches, dtype=np.int_)
    # matches = matches[matches[:, 0].argsort(), :]

    # get conflict matches
    conflicts_ind = list(
        set(np.arange(pre_matches.shape[0])) - set(matches_ind))
    conflict_matches = pre_matches[conflicts_ind, :]

    return list(map(tuple, matches)), conflict_matches


def pre_match_process(motion_results, appearance_results):
    # global pre_matches
    if motion_results[0].shape[0]: # 运动有匹配结果
        # 在这里if appearance_results[0].shape[0]不成立，使用运动，是否可以加pre_matches后面加一个1，
        # 如：pre_matches = [track_id, det_id, speed, num, 1]
        # pre_matches = np.vstack((motion_results[0], appearance_results[0])
        #                         ) if appearance_results[0].shape[0] else motion_results[0]
        # 添加运动flage
        if appearance_results[0].shape[0]:
            pre_matches = np.vstack((np.insert(motion_results[0],4,0,axis=1), np.insert(appearance_results[0],4,1,axis=1)))
        else:
            pre_matches = np.insert(motion_results[0],4,0,axis=1)

        unmatched_tracks = list(set(motion_results[1]).intersection(
            set(appearance_results[1])))
        unmatched_detections = list(set(motion_results[2]).intersection(
            set(appearance_results[2])))
    else:
        # 在这里是使用外观，是否可以在pre_matches后面加一个0，如：pre_matches = [track_id, det_id, speed，num, 0]
        pre_matches = appearance_results[0]
        # 添加
        if len(pre_matches) == 0:
            pass
        else:
            pre_matches = np.insert(pre_matches,4,1,axis=1)
        # pre_matches = np.insert(pre_matches,4,1)
        unmatched_tracks = appearance_results[1]
        unmatched_detections = appearance_results[2]

    return pre_matches, unmatched_tracks, unmatched_detections


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


def two_round_match(conflicts, alpha_gate):
    # conflicts 在match中没有的，冲突，如：match = [(0,0),[3,3]] conflicts = [1,1] [2,1]
    if conflicts.shape[0] == 0:
        return [], [], []
    # else:
    #     print("111111111111111111111111111")
    #     sys.exit()
    # appearance match
    # 使用外观，那么在matches_a后面添加0
    # first_round = conflicts[conflicts[:, -2] >= alpha_gate, :] # bee 0.9
    # (t_id, d_id, level, 0/1, uuid)

    first_round = conflicts[conflicts[:, 2] >= alpha_gate, :] # bee 0.9
    matches_a = first_round[first_round[:, 3] == 1, :]
    if len(matches_a) != 0:
        matches_a[:,4] = 2 # 这里表示如果有冲突，而且选择外观，那么第五列的值就变为2
    # matches_a = first_round[first_round[:, -1] == 1, :2]
    # print("matches_appr:",len(matches_a),end="\t")
    # motion match
    # second_round = conflicts[conflicts[:, -2] < alpha_gate, :]
    second_round = conflicts[conflicts[:, 2] < alpha_gate, :]
    second_round = second_round[second_round[:, 3] == 0, :]
    if len(second_round) != 0:
        second_round[:,4] = 3 # 这里表示如果有冲突，而且选择运动，那么第五列的值就变为3
    # mask = np.ones(second_round.shape[0]).reshape(-1, 1)
    # second_round = np.hstack((second_round, mask))

    # filter track
    second_round = select_pairs(second_round, 0, matches_a[:, 0])

    # filter det
    second_round = select_pairs(second_round, 1, matches_a[:, 1])
    # 使用运动，在matches_b后面添加1
    matches_b = second_round[:, :]
    # matches_b = second_round[:, :2]
    # print("matches_motion:",len(matches_b),end="\t")
    
    
    # match results
    matches_ = np.vstack((matches_a, matches_b))
    matches_ = matches_.astype(np.int_)
    unmatched_tracks_ = list(set(conflicts[:, 0]) - set(matches_[:, 0]))
    unmatched_tracks_ = [int(i) for i in unmatched_tracks_]
    unmatched_detections_ = list(set(conflicts[:, 1]) - set(matches_[:, 1]))
    unmatched_detections_ = [int(i) for i in unmatched_detections_]

    # print("TOPIC:",len(matches_),end="\t")
    # print("unmatched_tracks_:",len(unmatched_tracks_),end="\t")
    # print("unmatched_detections_:",len(unmatched_detections_),end="\n")
    return list(map(tuple, matches_)), unmatched_tracks_, unmatched_detections_


def filter_pairs(cost_matrix, gate):
    cost_matrix[cost_matrix > gate] = INFTY_COST

    return cost_matrix


def select_pairs(matches, col_id, pairs):
    print('select_pairs')
    for t in pairs: # appr track
        ind = np.argwhere(matches[:, col_id] == t)
        if len(ind):
            matches[ind[0], 3] = -1 # -1
    matches = matches[matches[:, 3] == 0, :] # 提取 motion track # 0

    return matches

import numpy as np
import scipy.spatial as sp
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
INFTY_COST = 999

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


def giou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) *
        (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) *
        (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])
    wc = xxc2 - xxc1
    hc = yyc2 - yyc1
    assert (wc > 0).all() and (hc > 0).all()
    area_enclose = wc * hc
    giou = iou - (area_enclose - wh) / area_enclose
    giou = (giou + 1.0) / 2.0  # resize from (-1,1) to (0,1)
    return giou


def diou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) *
        (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) *
        (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2
    diou = iou - inner_diag / outer_diag

    return (diou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ciou_batch(bboxes1, bboxes2):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    # calculate the intersection box
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    iou = wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) *
        (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) *
        (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    inner_diag = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    xxc1 = np.minimum(bboxes1[..., 0], bboxes2[..., 0])
    yyc1 = np.minimum(bboxes1[..., 1], bboxes2[..., 1])
    xxc2 = np.maximum(bboxes1[..., 2], bboxes2[..., 2])
    yyc2 = np.maximum(bboxes1[..., 3], bboxes2[..., 3])

    outer_diag = (xxc2 - xxc1) ** 2 + (yyc2 - yyc1) ** 2

    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    # prevent dividing over zero. add one pixel shift
    h2 = h2 + 1.0
    h1 = h1 + 1.0
    arctan = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / (np.pi**2)) * (arctan**2)
    S = 1 - iou
    alpha = v / (S + v)
    ciou = iou - inner_diag / outer_diag - alpha * v

    return (ciou + 1) / 2.0  # resize from (-1,1) to (0,1)


def ct_dist(bboxes1, bboxes2):
    """
    Measure the center distance between two sets of bounding boxes,
    this is a coarse implementation, we don't recommend using it only
    for association, which can be unstable and sensitive to frame rate
    and object speed.
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    centerx1 = (bboxes1[..., 0] + bboxes1[..., 2]) / 2.0
    centery1 = (bboxes1[..., 1] + bboxes1[..., 3]) / 2.0
    centerx2 = (bboxes2[..., 0] + bboxes2[..., 2]) / 2.0
    centery2 = (bboxes2[..., 1] + bboxes2[..., 3]) / 2.0

    ct_dist2 = (centerx1 - centerx2) ** 2 + (centery1 - centery2) ** 2

    ct_dist = np.sqrt(ct_dist2)

    # The linear rescaling is a naive version and needs more study
    ct_dist = ct_dist / ct_dist.max()
    return ct_dist.max() - ct_dist  # resize to (0,1)


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / \
        2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def compute_aw_new_metric(emb_cost, w_association_emb, max_diff=0.5):
    w_emb = np.full_like(emb_cost, w_association_emb)
    w_emb_bonus = np.full_like(emb_cost, 0)

    # Needs two columns at least to make sense to boost
    if emb_cost.shape[1] >= 2:
        # Across all rows
        for idx in range(emb_cost.shape[0]):
            inds = np.argsort(-emb_cost[idx])
            # Row weight is difference between top / second top
            row_weight = min(emb_cost[idx, inds[0]] -
                             emb_cost[idx, inds[1]], max_diff)
            # Add to row
            w_emb_bonus[idx] += row_weight / 2

    if emb_cost.shape[0] >= 2:
        for idj in range(emb_cost.shape[1]):
            inds = np.argsort(-emb_cost[:, idj])
            col_weight = min(emb_cost[inds[0], idj] -
                             emb_cost[inds[1], idj], max_diff)
            w_emb_bonus[:, idj] += col_weight / 2

    return w_emb + w_emb_bonus


def split_cosine_dist(dets, trks, affinity_thresh=0.55, pair_diff_thresh=0.6, hard_thresh=True):

    cos_dist = np.zeros((len(dets), len(trks)))

    for i in range(len(dets)):
        for j in range(len(trks)):

            # shape = 3x3
            cos_d = 1 - sp.distance.cdist(dets[i], trks[j], "cosine")
            patch_affinity = np.max(cos_d, axis=0)  # shape = [3,]
            # exp16 - Using Hard threshold
            if hard_thresh:
                if len(np.where(patch_affinity > affinity_thresh)[0]) != len(patch_affinity):
                    cos_dist[i, j] = 0
                else:
                    cos_dist[i, j] = np.max(patch_affinity)
            else:
                # can experiment with mean too (max works slightly better)
                cos_dist[i, j] = np.max(patch_affinity)

    return cos_dist

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

def cal_cost_matrix(dets_embs, trk_embs, metric):
    # features = np.array([d.feature for d in dets])
    # targets = np.array([t.id for t in trackers if t.hits >= min_hits])
    # cost_matrix = metric.distance(features, targets)
    print('metric: ', metric)
    if metric == 'res_recons':
    # 法1：外观重建，用外观更新
        cost_matrix, sim_matrix = _nn_res_recons_cosine_distance(
            trk_embs, dets_embs, data_is_normalized=False)
    else:
        # 法2：余弦
        cost_matrix, sim_matrix = _cosine_distance(trk_embs, dets_embs)
        print("cosine cost_matrix.min:%.2f, max:%.2f, mean:%.2f, std:%.2f" %
            (cost_matrix.min(), cost_matrix.max(), cost_matrix.mean(), cost_matrix.std()))
    return cost_matrix, sim_matrix
def filter_pairs(cost_matrix, gate):
    cost_matrix[cost_matrix > gate] = INFTY_COST

    return cost_matrix

def associate(
    detections,
    trackers,
    det_embs,
    trk_embs,
    # emb_cost,
    iou_threshold,
    velocities,
    previous_obs,
    vdc_weight,
    w_assoc_emb,
    aw_off,
    aw_param,
    emb_off,
    grid_off,
    track_indices, tracks_info, gate, metric
):
    if len(trackers) == 0:
        # !// 2、3参数顺序调换
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0, 5), dtype=int),
            np.arange(len(detections)),
        )
        # return (
        #     np.empty((0, 2), dtype=int),
        #     np.arange(len(detections)),
        #     np.empty((0, 5), dtype=int),
        # )

    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(
        detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    # !// 新增的计算emb_cost
    # emb_cost = None
    # if not emb_off:
    #     if grid_off:
    #         emb_cost = None if (
    #             trk_embs.shape[0] == 0 or det_embs.shape[0] == 0) else det_embs @ trk_embs.T
    #     else:
    #         emb_cost = split_cosine_dist(det_embs, trk_embs)
    # print("emb cost: \t max:%.2f; min:%.2f" % (emb_cost.max(), emb_cost.min()))
    # # 保存
    # import os
    # file_path = 'emb_cost.txt'
    # emb_avg, emb_std = emb_cost.mean(), emb_cost.std()
    # emb_max, emb_min = emb_cost.max(), emb_cost.min()
    # emb_save = np.asarray([emb_avg, emb_std, emb_max, emb_min]).reshape(1, -1)
    # if os.path.isfile(file_path):
    #     data = np.loadtxt(file_path)
    #     if data.ndim == 1:
    #         data = data.reshape(1, -1)
    #     emb_save = np.concatenate((data, emb_save), axis=0)
    # np.savetxt(file_path, emb_save)
    cost_matrix, emb_cost = cal_cost_matrix(det_embs, trk_embs, metric)

    # 用外观过滤运动的, 不用双轮使用下面代码
    # cost_matrix = filter_pairs(cost_matrix, gate)
    # cost_matrix[cost_matrix<INFTY_COST] = 1
    # cost_matrix[cost_matrix==INFTY_COST] = 0 # 过滤
    
    # cost_matrix = cost_matrix.T
    # emb_cost = emb_cost.T
    # iou_matrix = iou_matrix * cost_matrix

    # 用运动过滤外观
    # iou_matrix = filter_pairs(iou_matrix, gate)
    # iou_matrix[iou_matrix<INFTY_COST] = 1
    # iou_matrix[iou_matrix==INFTY_COST] = 0 # 过滤
    
    # cost_matrix = cost_matrix.T
    # emb_cost = emb_cost.T
    # iou_matrix = iou_matrix * cost_matrix



    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            # !// 新增的融入emb_cost的计算
            # !// ocsort中只是算了-(iou_matrix+angle_diff_cost)两项
            if emb_cost is None:
                emb_cost = 0
            else:
                # emb_cost[iou_matrix <= 0.3] = 0
                pass
            # print('w_assoc_emb1: ', w_assoc_emb)
            if not aw_off:
                w_matrix = compute_aw_new_metric(
                    emb_cost, w_assoc_emb, aw_param)
                emb_cost *= w_matrix
            else:
                emb_cost *= w_assoc_emb
            final_cost = -(iou_matrix + angle_diff_cost+emb_cost)
            # final_cost = -(iou_matrix + angle_diff_cost)
            # final_cost = -(iou_matrix + angle_diff_cost + emb_cost)
            matched_indices = linear_assignment(final_cost)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            # !// 引入运动强度
            track_idx = track_indices[m[1]]
            m_ = np.array([m[1], m[0], tracks_info[track_idx]])  # 匹配、运动强度
            matches.append(m_.reshape(1, 3))
            # matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        # !// 3列
        matches = np.empty((0, 3), dtype=int)
        # matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    # !// 缺少运动标记
    if matches.shape[0]:
        matches = np.hstack(
            (matches, np.zeros(matches.shape[0]).reshape(-1, 1)))

    # !// 2、3参数顺序调换
    return (matches, np.array(unmatched_trackers), np.array(unmatched_detections))
    # return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def associate_kitti(detections, trackers, det_cates, iou_threshold, velocities, previous_obs, vdc_weight):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    """
        Cost from the velocity direction consistency
    """
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    scores = np.repeat(
        detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    """
        Cost from IoU
    """
    iou_matrix = iou_batch(detections, trackers)

    """
        With multiple categories, generate the cost for catgory mismatch
    """
    num_dets = detections.shape[0]
    num_trk = trackers.shape[0]
    cate_matrix = np.zeros((num_dets, num_trk))
    for i in range(num_dets):
        for j in range(num_trk):
            if det_cates[i] != trackers[j, 4]:
                cate_matrix[i][j] = -1e6

    cost_matrix = -iou_matrix - angle_diff_cost - cate_matrix

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(cost_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

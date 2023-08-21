"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import pdb
import pickle

import cv2
import torch
import torchvision

import numpy as np
# from .association import *
from .association_yolo import *
from .embedding import EmbeddingComputer
from .cmc import CMCComputer
from .assignment import *
from .nn_matching import *


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_bbox_to_z_new(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    return np.array([x, y, w, h]).reshape((4, 1))


def convert_x_to_bbox_new(x):
    x, y, w, h = x.reshape(-1)[:4]
    return np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]).reshape(1, 4)


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score == None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6

    level = 1-iou_batch(np.array([bbox1]), np.array([bbox2]))[0][0]
    return speed / norm, level


def new_kf_process_noise(w, h, p=1 / 20, v=1 / 160):
    Q = np.diag(
        (
            (p * w) ** 2,
            (p * h) ** 2,
            (p * w) ** 2,
            (p * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
            (v * w) ** 2,
            (v * h) ** 2,
        )
    )
    return Q


def new_kf_measurement_noise(w, h, m=1 / 20):
    w_var = (m * w) ** 2
    h_var = (m * h) ** 2
    R = np.diag((w_var, h_var, w_var, h_var))
    return R


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, delta_t=3, orig=False, emb=None, alpha=0, new_kf=False):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        if not orig:
            from .kalmanfilter import KalmanFilterNew as KalmanFilter
        else:
            from filterpy.kalman import KalmanFilter

        self.new_kf = new_kf
        if new_kf:
            self.kf = KalmanFilter(dim_x=8, dim_z=4)
            self.kf.F = np.array(
                [
                    # x y w h x' y' w' h'
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                ]
            )
            _, _, w, h = convert_bbox_to_z_new(bbox).reshape(-1)
            self.kf.P = new_kf_process_noise(w, h)
            self.kf.P[:4, :4] *= 4
            self.kf.P[4:, 4:] *= 100
            # Process and measurement uncertainty happen in functions
            self.bbox_to_z_func = convert_bbox_to_z_new
            self.x_to_bbox_func = convert_x_to_bbox_new
        else:
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [
                    # x  y  s  r  x' y' s'
                    [1, 0, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            # give high uncertainty to the unobservable initial velocities
            self.kf.P[4:, 4:] *= 1000.0
            self.kf.P *= 10.0
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[4:, 4:] *= 0.01
            self.bbox_to_z_func = convert_bbox_to_z
            self.x_to_bbox_func = convert_x_to_bbox

            # Attempt
            # self.kf.P[2, 2] = 10000
            # self.kf.R[2, 2] = 10000

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder # TODO: 尝试给bbox
        # Used to output track after min_hits reached
        self.history_observations = []
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.speed = 1 # TODO：初始化速度
        self.delta_t = delta_t

        self.emb = emb

        self.frozen = False

        self.budget = 30  # 外观库的大小
        self.emb_ind = 0
        # self.emb_bank = np.zeros((10000, 2048))
        # self.emb_bank[self.emb_ind, :] = self.emb

        self.emb_ind += 1

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        if bbox is not None:
            self.frozen = False

            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                """
                  Estimate the track speed direction with observations \Delta t steps away
                """
                self.velocity, self.speed = speed_direction(previous_box, bbox)
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            if self.new_kf:
                R = new_kf_measurement_noise(self.kf.x[2, 0], self.kf.x[3, 0])
                self.kf.update(self.bbox_to_z_func(bbox), R=R, new_kf=True)
            else:
                self.kf.update(self.bbox_to_z_func(bbox))
        else:
            self.kf.update(bbox, new_kf=self.new_kf)
            self.frozen = True

    def update_emb(self, emb, alpha=0.9):
        # self.emb_bank[self.emb_ind, :] = emb
        # self.emb_ind += 1

        # 官方
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        # start_ind = max(0, self.emb_ind-self.budget)
        # return self.emb_bank[start_ind:self.emb_ind]
        return self.emb

    def apply_affine_correction(self, affine):
        m = affine[:, :2]
        t = affine[:, 2].reshape(2, 1)
        # For OCR
        if self.last_observation.sum() > 0:
            ps = self.last_observation[:4].reshape(2, 2).T
            ps = m @ ps + t
            self.last_observation[:4] = ps.T.reshape(-1)

        # Apply to each box in the range of velocity computation
        for dt in range(self.delta_t, -1, -1):
            if self.age - dt in self.observations:
                ps = self.observations[self.age - dt][:4].reshape(2, 2).T
                ps = m @ ps + t
                self.observations[self.age - dt][:4] = ps.T.reshape(-1)

        # Also need to change kf state, but might be frozen
        self.kf.apply_affine_correction(m, t, self.new_kf)

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if self.new_kf:
            if self.kf.x[2] + self.kf.x[6] <= 0:
                self.kf.x[6] = 0
            if self.kf.x[3] + self.kf.x[7] <= 0:
                self.kf.x[7] = 0

            # Stop velocity, will update in kf during OOS
            if self.frozen:
                self.kf.x[6] = self.kf.x[7] = 0
            Q = new_kf_process_noise(self.kf.x[2, 0], self.kf.x[3, 0])
        else:
            if (self.kf.x[6] + self.kf.x[2]) <= 0:
                self.kf.x[6] *= 0.0
            Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        # 当前没被跟踪到就归零，要再连续跟踪到3帧才能被匹配
        if self.time_since_update > 0:
            self.hit_streak = 0 
        self.time_since_update += 1
        # TODO：未来可以对历史轨迹做一个处理
        self.history.append(self.x_to_bbox_func(self.kf.x)) # 历史轨迹
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class OCSort(object):
    def __init__(
        self,
        det_thresh,
        alpha_gate,
        gate,
        gate2,
        # max_age=30,
        max_age=15,
        min_hits=0, # TODO: 可以调参为0
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        w_association_emb=0.75,
        alpha_fixed_emb=0.95,
        aw_param=0.5,
        embedding_off=False,
        cmc_off=False,
        aw_off=False,
        new_kf_off=False,
        grid_off=False,
        dynamic_appr_off=False,

        **kwargs,
    ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        KalmanBoxTracker.count = 0

        self.embedder = EmbeddingComputer(
            kwargs["args"].dataset, kwargs["args"].test_dataset, grid_off)
        self.cmc = CMCComputer()
        self.embedding_off = embedding_off
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        self.new_kf_off = new_kf_off
        self.grid_off = grid_off
        self.dynamic_appr_off = dynamic_appr_off
        self.min_hits = min_hits

        self.sigma = 5  # 运动强度计算参数(已失效)

        # TODO: 改进点：看下每个序列目标的历史运动轨迹的波动状态，根据历史状态来决定用外观还是运动特征，而不是只根据前一帧的位移

        self.alpha_gate = alpha_gate  # 双轮匹配的运动强度阈值 【0.3~0.7】
        self.gate = gate  # 外观阈值 0.4
        self.gate2 = gate2  # 外观阈值 0.4

        """
        输出：横坐标：时间，纵坐标：运动强度
        """

    def map_scores(self, dets):
        # 将得分映射到一个范围为0到1之间的值，表示检测器对检测结果的置信程度。
        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = self.alpha_fixed_emb  # 0.95
        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident
        # 当检测器对检测结果的置信度较低时，dets_alpha的值趋近于1，表示对嵌入特征的信任度较高。
        # 动态外观DA
        if not self.dynamic_appr_off:
            # print('use da')
            dets_alpha = af + (1 - af) * (1 - trust)
        else:
            dets_alpha = af * np.ones_like(trust)

        return dets_alpha

    def extract_detections(self, output_results, img_tensor, img_numpy, tag = None):
        # 从output_results中提取score和bbox信息，并将它们合并为一个dets数组，最后返回这个数组。
        if not isinstance(output_results, np.ndarray):
            output_results = output_results.cpu().numpy()

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        
        dets = np.concatenate(
            (bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        
        # Rescale
        scale = min(img_tensor.shape[2] / img_numpy.shape[0],
                    img_tensor.shape[3] / img_numpy.shape[1])

        dets[:, :4] /= scale
        
        # low det
        inds_low = scores > 0.4
        inds_high = scores <= self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        
        # high det
        remain_inds = scores > self.det_thresh  # 0.6
        dets_one = dets[remain_inds]
        frame_id = tag.split(':')[1]
        video_name = tag.split(':')[0]
        # for i in range(len(dets)):
        #     det = dets[i]
        #     det_info = [frame_id, 1, det[0], det[1], det[2]-det[0], det[3] - det[1], det[4]]
        #     record = ','.join(list(map(str, det_info)))
        #     with open('results/gmot_det/' + video_name + '.txt', "a+") as f:
        #         f.write(record+"\n")
        #     print('det: ', det_info)


        # print('det11111: ', dets)

        return dets_one, dets_second

    def generate_embs(self, dets, img_numpy, tag):
        dets_embs = np.ones((dets.shape[0], 1))
        
        if not self.embedding_off and dets.shape[0] != 0:
            # print("use emb")
            # Shape = (num detections, 3, 512) if grid
            dets_embs = self.embedder.compute_embedding(
                img_numpy, dets[:, :4], tag)
        return dets_embs

    def get_pred_loc_from_exist_tracks(self):
        trks = np.zeros((len(self.trackers), 5))
        trk_embs = []
        to_del = []

        # max_ind = max(
        #     self.trackers[t].emb_ind for t in range(len(self.trackers)))
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.trackers[t].get_emb())
                # trk_embs.append(self.trackers[t].get_emb(max_ind))
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        # Shape = (num_trackers, 3, 512) if grid
        trk_embs = np.array(trk_embs)  # TODO: 为了变长轨迹
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array(
            (0, 0)) for trk in self.trackers])
        speeds = np.array(
            [trk.speed if trk.speed is not None else 0 for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(
            trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        return trks, velocities, speeds, last_boxes, k_observations, trk_embs

    def motion_third_associate(self,  dets, last_boxes, unmatched_dets, unmatched_trks, tracks_info,iou_threshold):
        left_dets = dets[unmatched_dets]
        left_trks = last_boxes[unmatched_trks]

        # TODO: maybe use embeddings here
        iou_left = self.asso_func(left_dets, left_trks)
        iou_left = np.array(iou_left)
        # !// 匹配结果保存
        matches = []
        if iou_left.max() > iou_threshold:
            """
            NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
            get a higher performance especially on MOT17/MOT20 datasets. But we keep it
            uniform here for simplicity
            """
            rematched_indices = linear_assignment(-iou_left)

            to_remove_det_indices = []
            to_remove_trk_indices = []
            for m in rematched_indices:
                det_ind, trk_ind = unmatched_dets[m[0]
                                                  ], unmatched_trks[m[1]]
                if iou_left[m[0], m[1]] < iou_threshold:
                    continue

                # !// 不在这里更新
                # self.trackers[trk_ind].update(dets[det_ind, :])
                # self.trackers[trk_ind].update_emb(
                #     dets_embs[det_ind], alpha=dets_alpha[det_ind])

                # !// 运动强度
                m_ = np.array([trk_ind, det_ind, tracks_info[trk_ind]])
                # m_ = np.array([m[1], m[0], tracks_info[trk_ind]])
                matches.append(m_.reshape(1, 3))

                to_remove_det_indices.append(det_ind)
                to_remove_trk_indices.append(trk_ind)
            unmatched_dets = np.setdiff1d(
                unmatched_dets, np.array(to_remove_det_indices))
            unmatched_trks = np.setdiff1d(
                unmatched_trks, np.array(to_remove_trk_indices))

        # !// 匹配新增列
        matches = np.array(matches)
        if (len(matches) == 0):
            matches = np.empty((0, 3), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        matches = np.hstack(
            (matches, np.zeros(matches.shape[0]).reshape(-1, 1)))

        return (matches, unmatched_trks, unmatched_dets)

    def update(self, output_results, img_tensor, img_numpy, tag, metric, two_round_off):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """

        if output_results is None:
            return np.empty((0, 5))
        self.frame_count += 1
        
        # 获取det和emb
        dets_one, dets_second = self.extract_detections(output_results, img_tensor, img_numpy, tag)
        
        dets_one_embs = self.generate_embs(dets_one, img_numpy, tag+"@one")
        
        dets_second_embs = self.generate_embs(dets_second, img_numpy, tag+"@second")
        # print('dets_second2: ', dets_second_embs)
        # # CMC
        # if not self.cmc_off:
        #     print("use cmc")
        #     transform = self.cmc.compute_affine(img_numpy, dets_one[:, :4], tag)
        #     for trk in self.trackers:
        #         trk.apply_affine_correction(transform)

        # 得分映射
        dets_one_alpha = self.map_scores(dets_one)
        dets_second_alpha = self.map_scores(dets_second)
        print("dets one: %d" % (len(dets_one_alpha)),end="\t")
        print("dets second: %d" % (len(dets_second_alpha)))

        # 匹配
        ret = self.assign_cascade(dets_one, dets_one_embs, dets_one_alpha,
                                  dets_second,dets_second_embs,dets_second_alpha, metric, two_round_off)
        # ret = self.assign_(dets_one, dets_one_embs, dets_one_alpha)

        return ret


    def level_matching(self,depth,
                       track_indices, 
                       detection_one_indices,
                       detection_second_indices,
                        trk_embs,
                       dets_one_embs,
                       dets_second_embs,
                        trks,
                        dets_one, 
                        dets_second, 
                        speeds,
                        velocities,
                        k_observations,
                        last_boxes,
                        metric,
                        two_round_off,
                ):

        if track_indices is None:
            track_indices = np.arange(len(trks))
        if detection_one_indices is None:
            detection_one_indices = np.arange(len(dets_one))
        if detection_second_indices is None:
            detection_second_indices = np.arange(len(dets_second))
            
        """
            First round of association
        """
        if len(track_indices):
            # 获取基于高斯核的运动强度
            tracks_info = {}
            alpha = metric_gaussian_motion(speeds, sigma=self.sigma)
            for i, idx in enumerate(track_indices):
                tracks_info[idx] = alpha[i]
        if len(detection_one_indices) != 0 and len(track_indices) != 0:
            # 外观预匹配
            gate = self.gate if depth==0 else self.gate2
            appearance_pre_assign, emb_cost = appearance_associate(
                dets_one_embs,
                trk_embs,
                dets_one, trks,
                track_indices, detection_one_indices, tracks_info,
                gate, self.iou_threshold, metric)
            
            # 运动预匹配
            # if depth ==0:
            #     motion_pre_assign = associate(
            #         dets_one,
            #         trks,
            #         dets_one_embs,
            #         trk_embs,
            #         # emb_cost,
            #         self.iou_threshold,
            #         velocities,
            #         k_observations,
            #         self.inertia,
            #         self.w_association_emb,
            #         self.aw_off,
            #         self.aw_param,
            #         self.embedding_off,
            #         self.grid_off,
            #         track_indices, tracks_info, gate, metric
            #     )
            # else:
            #     motion_pre_assign = (np.empty(
            #     (0, 4), dtype=int),[],[])
            
            # 外观消融_不加双轮使用以下代码，并注释上面appearance_associate
            # appearance_pre_assign = motion_pre_assign

            # 运动过滤外观
            motion_pre_assign = appearance_pre_assign

            # 双轮匹配
           
            matched_one_1, unmatched_trks_1, unmatched_dets_one_1 = min_cost_matching(
                motion_pre_assign, appearance_pre_assign, self.alpha_gate, two_round_off)
        else:
           
            matched_one_1, unmatched_trks_1, unmatched_dets_one_1 = np.empty(
                (0, 2), dtype=int), track_indices, detection_one_indices
        
        """
            Second round of associaton by OCR
        """
        unmatched_trks = unmatched_trks_1
        if len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            iou_threshold = 0.9
            # gate = self.gate if depth==0 else self.gate2
            gate = self.gate2
    
            if depth==0:
                # 法1
                # 用IOU匹配
                motion_pre_assign = self.motion_second_associate(
                    trks, dets_second, unmatched_trks, tracks_info,iou_threshold)
                
                appearance_pre_assign =motion_pre_assign
                
                # 法2：
                # left_trk_embs = [trk_embs[i] for i in unmatched_trks]
                # left_trks = trks[unmatched_trks]
                # appearance_pre_assign,_ = appearance_associate(
                    # dets_second_embs,
                    # left_trk_embs,
                    # dets_second,left_trks,
                    # unmatched_trks, detection_second_indices,
                    # tracks_info,
                    # gate, self.iou_threshold)
                # motion_pre_assign=appearance_pre_assign
                
                # 法3：
                # motion_pre_assign = self.motion_second_associate(
                #     trks, dets_second, unmatched_trks, tracks_info,iou_threshold)
                # left_trk_embs = [trk_embs[i] for i in unmatched_trks]
                # left_trks = trks[unmatched_trks]
                # appearance_pre_assign,_ = appearance_associate(
                #     dets_second_embs,
                    # left_trk_embs,
                    # dets_second,left_trks,
                    # unmatched_trks, detection_second_indices,
                    # tracks_info,
                    # gate, self.iou_threshold)
                
            else:   
                left_trk_embs = [trk_embs[i] for i in unmatched_trks]
                left_trks = trks[unmatched_trks]
                
                appearance_pre_assign,_ = appearance_associate(
                    dets_second_embs, 
                    left_trk_embs,
                    dets_second,left_trks,
                    unmatched_trks, detection_second_indices,
                    tracks_info,
                    gate, self.iou_threshold)
                motion_pre_assign = appearance_pre_assign
        
            matches_second, unmatched_trks_second, unmatched_dets_second = min_cost_matching(
                motion_pre_assign, appearance_pre_assign, self.alpha_gate)
        else:
         
            matches_second= np.empty((0, 2), dtype=int)
            unmatched_trks_second = unmatched_trks
            unmatched_dets_second = detection_second_indices
            
        unmatched_trks_2 = unmatched_trks_second
        # 处理高score的未匹配的dets
        if unmatched_dets_one_1.shape[0] > 0 and unmatched_trks_2.shape[0] > 0:
            iou_threshold = self.iou_threshold
            # gate = self.gate if depth==0 else self.gate2
            gate = self.gate2
            if depth==0:
                # 仅用IOU进行匹配
                motion_pre_assign = self.motion_third_associate(
                    dets_one, last_boxes, unmatched_dets_one_1, unmatched_trks_2, tracks_info,iou_threshold)
                
                # 法1：
                appearance_pre_assign = motion_pre_assign
                
                # 法2：
                # # 外观预匹配
                # left_dets_embs = dets_one_embs[unmatched_dets_one_1]
                # left_trks_embs = [trk_embs[i] for i in unmatched_trks_2]
                # left_dets = dets_one[unmatched_dets_one_1]
                # left_trks = trks[unmatched_trks_2]                
                # appearance_pre_assign, _ = appearance_associate(
                #     left_dets_embs,
                #     left_trks_embs,
                #     left_dets, left_trks,
                #     unmatched_trks_2, unmatched_dets_one_1,
                #     tracks_info,
                #     gate, iou_threshold)
                
                # # 外观预匹配
                # left_dets_embs = dets_one_embs[unmatched_dets_one_1]
                # left_trks_embs = [trk_embs[i] for i in unmatched_trks_2]
                # left_dets = dets_one[unmatched_dets_one_1]
                # left_trks = trks[unmatched_trks_2]                
                # appearance_pre_assign, _ = appearance_associate(
                #     left_dets_embs,
                #     left_trks_embs,
                #     left_dets, left_trks,
                #     unmatched_trks_2, unmatched_dets_one_1,
                #     tracks_info,
                #     gate, iou_threshold,rotate=True)

                # motion_pre_assign = appearance_pre_assign

            else:
                # 外观预匹配
                left_dets_embs = dets_one_embs[unmatched_dets_one_1]
                left_trks_embs = [trk_embs[i] for i in unmatched_trks_2]
                left_dets = dets_one[unmatched_dets_one_1]
                left_trks = trks[unmatched_trks_2]                
                appearance_pre_assign, _ = appearance_associate(
                    left_dets_embs,
                    left_trks_embs,
                    left_dets, left_trks,
                    unmatched_trks_2, unmatched_dets_one_1,
                    tracks_info,
                    gate, iou_threshold)

                motion_pre_assign = appearance_pre_assign
            # 双轮匹配
           
            matched_one_2, unmatched_trks_3, unmatched_dets_one_2 = min_cost_matching(
                motion_pre_assign, appearance_pre_assign, self.alpha_gate)
        else:
         
            matched_one_2 = np.empty((0, 2), dtype=int)
            unmatched_trks_3=unmatched_trks_2
            unmatched_dets_one_2=unmatched_dets_one_1
            
        if len(matched_one_1) and len(matched_one_2):
            matches_one = np.concatenate((matched_one_1,matched_one_2),axis=0)
        elif len(matched_one_2):
            matches_one = matched_one_2
        elif len(matched_one_1):
            matches_one = matched_one_1
        else:
            matches_one = np.empty((0, 2), dtype=int)
            
        unmatched_trks = unmatched_trks_3
        unmatched_dets_one = unmatched_dets_one_2
        
        
        return matches_one,matches_second, unmatched_trks, unmatched_dets_one,unmatched_dets_second

    def motion_second_associate(self, trks, dets_second, unmatched_trks, tracks_info,iou_threshold):
        u_trks = trks[unmatched_trks]
       
        iou_left = self.asso_func(dets_second, u_trks)
        iou_left = np.array(iou_left)

        matched_2, unmatched_trks_2 = [], unmatched_trks
        if iou_left.max() > iou_threshold:
           

            matched_indices = linear_assignment(-iou_left)

            to_remove_trk_indices = []
            for m in matched_indices:
                det_ind, trk_ind = m[0], unmatched_trks[m[1]]

                if iou_left[m[0], m[1]] < iou_threshold:
                    continue
                m_ = np.array([trk_ind, det_ind, tracks_info[trk_ind]])
                matched_2.append(m_.reshape(1, 3))
               
                to_remove_trk_indices.append(trk_ind)
          
            unmatched_trks_2 = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        matched_2 = np.array(matched_2)
        if(len(matched_2) == 0):
            matched_2 = np.empty((0, 3), dtype=int)
        else:
            matched_2 = np.concatenate(matched_2, axis=0)

        matched_2 = np.hstack(
            (matched_2, np.zeros(matched_2.shape[0]).reshape(-1, 1)))

        return (matched_2, unmatched_trks_2, np.array([]))
    
    def level_matching_v0(self,depth,
                       track_indices, detection_indices,
                        trk_embs,
                       dets_embs,
                        trks,
                        dets, 
                        dets_alpha,
                        speeds,
                        velocities,
                        k_observations,
                        last_boxes,
                ):

        if track_indices is None:
            track_indices = np.arange(len(trks))
        if detection_indices is None:
            detection_indices = np.arange(len(dets))
            
        if len(detection_indices) == 0 or len(track_indices) == 0:
            return np.empty((0, 2), dtype=int), track_indices, detection_indices # Nothing to match.
        """
            First round of association
        """
        if len(detection_indices) != 0 and len(track_indices) != 0:
            # 获取基于高斯核的运动强度
            tracks_info = {}
            alpha = metric_gaussian_motion(speeds, sigma=self.sigma)
            for i, idx in enumerate(track_indices):
                tracks_info[idx] = alpha[i]
                # print("speed:", alpha[i])

            # 外观预匹配
            gate = self.gate if depth==0 else self.gate2
            appearance_pre_assign, emb_cost = appearance_associate(
                dets_embs,
                trk_embs,
                dets, trks,
                track_indices, detection_indices, tracks_info,
                gate, self.iou_threshold)
            
            # 运动预匹配
            if depth ==0:
                motion_pre_assign = associate(
                    dets,
                    trks,
                    dets_embs,
                    trk_embs,
                    emb_cost,
                    self.iou_threshold,
                    velocities,
                    k_observations,
                    self.inertia,
                    self.w_association_emb,
                    self.aw_off,
                    self.aw_param,
                    self.embedding_off,
                    self.grid_off,
                    track_indices, tracks_info,
                )
            else:
                motion_pre_assign = (np.empty(
                (0, 4), dtype=int),[],[])

            # 双轮匹配
            matched_1, unmatched_trks_1, unmatched_dets_1 = min_cost_matching(
                motion_pre_assign, appearance_pre_assign, self.alpha_gate)
        else:
            matched_1, unmatched_trks_1, unmatched_dets_1 = np.empty(
                (0, 2), dtype=int), np.empty((0, 5), dtype=int), detection_indices
            
        """
            Second round of associaton by OCR
        """
        if unmatched_dets_1.shape[0] > 0 and unmatched_trks_1.shape[0] > 0:
            # motion_pre_assign = self.motion_third_associate(
            #         dets, last_boxes, unmatched_dets_1, unmatched_trks_1, tracks_info)
            if depth==0:
                motion_pre_assign = self.motion_third_associate(
                    dets, last_boxes, unmatched_dets_1, unmatched_trks_1, tracks_info)
            else:
                motion_pre_assign = (np.empty(
                (0, 4), dtype=int),[],[])
            # 外观预匹配
            left_dets_embs = dets_embs[unmatched_dets_1]
            left_trks_embs = [trk_embs[i] for i in unmatched_trks_1]
            left_dets = dets[unmatched_dets_1]
            left_trks = trks[unmatched_trks_1]
            # left_trks_embs = trk_embs[unmatched_trks_1]
            gate = self.gate if depth==0 else self.gate2
            appearance_pre_assign, emb_cost = appearance_associate(
                left_dets_embs,
                left_trks_embs,
                left_dets, left_trks,
                unmatched_trks_1, unmatched_dets_1,
                tracks_info,
                gate, self.iou_threshold)
            # 双轮匹配
            matched_2, unmatched_trks_2, unmatched_dets_2 = min_cost_matching(
                motion_pre_assign, appearance_pre_assign, self.alpha_gate)
        else:
            matched_2 = np.empty((0, 2), dtype=int)
            unmatched_trks_2=unmatched_trks_1
            unmatched_dets_2=unmatched_dets_1
        # matched_2 = np.empty((0, 2), dtype=int)
        # unmatched_trks_2=unmatched_trks_1
        # unmatched_dets_2=unmatched_dets_1
            
            
        if len(matched_1) and len(matched_2):
            matched = np.concatenate((matched_1,matched_2),axis=0)
        elif len(matched_2):
            matched = matched_2
        elif len(matched_1):
            matched = matched_1
        else:
            matched = np.empty((0, 2), dtype=int)
        unmatched_trks = unmatched_trks_2
        unmatched_dets = unmatched_dets_2
        
        return matched, unmatched_trks, unmatched_dets

    def map_origin_ind(self, track_indices_l,
        unmatched_dets,
        matches_l, unmatched_trks_l, unmatched_dets_l):
        
        track_indices_l = np.array(track_indices_l)
        unmatched_dets = np.array(unmatched_dets)
        
        # matches
        if matches_l.shape[0]:
            matches_l_ = matches_l[:,:2].astype(np.int_)
            matches_l[:,0] = track_indices_l[matches_l_[:,0]]
            matches_l[:,1] = unmatched_dets[matches_l_[:,1]]
        
        # unmatched_trks
        if len(unmatched_trks_l):
            unmatched_trks_l[:] = track_indices_l[unmatched_trks_l[:]]
        
        # unmatched_dets
        if len(unmatched_dets_l):
            unmatched_dets_l[:] = unmatched_dets[unmatched_dets_l[:]]
        
        return matches_l, unmatched_trks_l, unmatched_dets_l
    
    def assign_cascade(self, dets_one, dets_one_embs, dets_one_alpha,
                                  dets_second,dets_second_embs,dets_second_alpha, metric, two_round_off):
        ret = []
        # get predicted locations from existing trackers.
        # 获取当前所有轨迹的信息
        trks, velocities, speeds, last_boxes, k_observations, trk_embs = \
            self.get_pred_loc_from_exist_tracks()

        
        # if self.frame_count == 5:
        #     print(1)
        """级联-双轮匹配"""
        track_indices = list(range(len(trks)))
        detection_one_indices = list(range(len(dets_one)))
        detection_second_indices = list(range(len(dets_second)))
        unmatched_dets_one = detection_one_indices
        
        # 取消low det
        # detection_second_indices = []
        unmatched_dets_second = detection_second_indices

        # 修改
        matched_one = np.empty((0, 5), dtype=int)
        matched_second = np.empty((0, 5), dtype=int)
        # matched_one = np.empty((0, 2), dtype=int)
        # matched_second = np.empty((0, 2), dtype=int)



        unmatched_trks_l = []
        unmatched_trks_final = []
        # TODO: 改为trades的方案
        for depth in range(1):
        # for depth in range(self.max_age):
            if len(unmatched_dets_one) == 0 and len(unmatched_dets_second)==0:  # No detections left
                break
            # 获取当前level的trk 索引号
            
            # 一次性匹配
            track_indices_l = [
            k for k in track_indices if self.trackers[k].time_since_update >= 1 + depth]
            
            # # 两次匹配
            # if depth > 0:
            #     track_indices_l = [
            #     k for k in track_indices if self.trackers[k].time_since_update >= 1 + depth]
            # else:    
            #     track_indices_l = [
            #         k for k in track_indices if self.trackers[k].time_since_update == 1 + depth]
            
            # 级联匹配
            # track_indices_l = [
            # k for k in track_indices if self.trackers[k].time_since_update == 1 + depth]
            
            if len(track_indices_l) == 0:  # Nothing to match at this depth
                continue
            # track_indices_l += list(unmatched_trks_l)
            # # 每一轮匹配完剩下的trk
            # unmatched_trks_final += list(unmatched_trks_l)
            # if depth == self.max_age:
            #     track_indices_l = unmatched_trks_final
            # if depth > 0:
            #     print(1)
            matches_one_l,matches_second_l, unmatched_trks_l, unmatched_dets_one_l,unmatched_dets_second_l = \
                self.level_matching(depth, None,None, None,
                                trk_embs[track_indices_l],
                                dets_one_embs[unmatched_dets_one],
                                dets_second_embs[unmatched_dets_second],
                                trks[track_indices_l],
                                dets_one[unmatched_dets_one], 
                                dets_second[unmatched_dets_second], 
                                speeds[track_indices_l],
                                velocities[track_indices_l],
                                k_observations[track_indices_l],
                                last_boxes[track_indices_l], metric, two_round_off)
            # 映射ind
            matches_one_l, unmatched_trks_l, unmatched_dets_one_l = \
                self.map_origin_ind(track_indices_l,
                unmatched_dets_one,
                matches_one_l, unmatched_trks_l, unmatched_dets_one_l)
            matches_second_l, _, unmatched_dets_second_l = \
                self.map_origin_ind(track_indices_l,
                unmatched_dets_second,
                matches_second_l, [], unmatched_dets_second_l)
            
            # 更新
            unmatched_dets_one = list(unmatched_dets_one_l)
            unmatched_dets_second = list(unmatched_dets_second_l)
            if len(matches_one_l):
                matched_one = np.concatenate((matched_one, matches_one_l),axis=0)
            if len(matches_second_l):
                matched_second = np.concatenate((matched_second, matches_second_l),axis=0)
        
        unmatched_trks = list(set(track_indices) - set(matched_one[:,0])- set(matched_second[:,0]))
        
        
        """后处理"""
        # 更新轨迹
        for m in matched_one:
            self.trackers[int(m[0])].update(dets_one[int(m[1]), :])
            self.trackers[int(m[0])].update_emb(
                dets_one_embs[int(m[1])], alpha=dets_one_alpha[int(m[1])])
        
        # 禁用low det
        for m in matched_second:
            self.trackers[int(m[0])].update(dets_second[int(m[1]), :])
            self.trackers[int(m[0])].update_emb(
                dets_second_embs[int(m[1])], alpha=dets_second_alpha[int(m[1])])
            
        for m in unmatched_trks:
            self.trackers[m].update(None)
            
        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets_one:
            trk = KalmanBoxTracker(
                dets_one[i, :], delta_t=self.delta_t, emb=dets_one_embs[i], alpha=dets_one_alpha[i], new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)
        
        for i in unmatched_dets_second:
            trk = KalmanBoxTracker(
                dets_second[i, :], delta_t=self.delta_t, emb=dets_second_embs[i], alpha=dets_second_alpha[i], new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)
        
        # 提取当前帧跟踪的结果
        # ret = self.final_process()
        ret = self.final_process(matched_one)
        
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
    def get_row_by_id(self, array, id):
        for row in array:
            if row[0] == id:
                return row[4]
        return None


    def final_process(self,matched_one):
        i = len(self.trackers)
        global flage_arr_mon
        global motion
        ret = []
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            # 当前跟踪到的 & (连续min_hits次被跟踪到 or 视频帧号<=min_hits)
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                #TODO: 在ret每一个目标最后加上一个外观运动标志（0/1）,该标识符可以在matched_one获取（已经修改完的）
                # ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
                
                if len(matched_one) != 0:
                    print('d',d)
                    print("id： ", [trk.id + 1])
                    flage = matched_one[matched_one[:,0] == int(trk.id),:]
                    if len(flage) != 0:
                        motion = flage[0][2]
                        flage_arr_mon = flage[0][4]
                    else:
                        motion = 1
                        flage_arr_mon = 1
                    # for row in matched_one:
                    #     if row[0] == int(trk.id):
                    #         # TODO: flage_arr_mon = row[2] # speed level
                    #         motion = row[2]
                    #         flage_arr_mon = row[4] # method, 0:m, 1:a
                    #     else:
                    #         motion = 1
                    #         flage_arr_mon = 1
                    ret.append(np.concatenate((d, [trk.id + 1],np.array([flage_arr_mon]),np.array([motion]))).reshape(1, -1))
                else:
                    ret.append(np.concatenate((d, [trk.id + 1], np.array([0]),np.array([1]))).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
            
        return ret

    def assign_(self, dets, dets_embs, dets_alpha):
        ret = []
        # get predicted locations from existing trackers.
        trks, velocities, speeds, last_boxes, k_observations, trk_embs = \
            self.get_pred_loc_from_exist_tracks()

        """
            First round of association
        """
        # !// 无此项
        # 获取首次匹配
        track_indices = list(range(len(trks)))
        detection_indices = list(range(len(dets)))
        # print('track_len: ', trks)
        print("dets_len: ", len(dets))
        if len(detection_indices) != 0 and len(track_indices) != 0:
            # 获取基于高斯核的运动强度
            tracks_info = {}
            alpha = metric_gaussian_motion(speeds, sigma=self.sigma)
            for i, idx in enumerate(track_indices):
                tracks_info[idx] = alpha[i]
                # print("speed:", alpha[i])

            # 外观预匹配
            appearance_pre_assign, emb_cost = appearance_associate(
                dets_embs,
                trk_embs,
                dets, trks,
                track_indices, detection_indices, tracks_info,
                self.gate, self.iou_threshold)
            print("pre_app: ", appearance_pre_assign)
            # 运动预匹配
            # motion_pre_assign = (matched, unmatched_trks, unmatched_dets)
            motion_pre_assign = associate(
                dets,
                trks,
                dets_embs,
                trk_embs,
                emb_cost,
                self.iou_threshold,
                velocities,
                k_observations,
                self.inertia,
                self.w_association_emb,
                self.aw_off,
                self.aw_param,
                self.embedding_off,
                self.grid_off,
                track_indices, tracks_info,
            )
            print("pre_motion: ", motion_pre_assign)

            # 双轮匹配

            matched, unmatched_trks, unmatched_dets = min_cost_matching(
                motion_pre_assign, appearance_pre_assign, self.alpha_gate)
            # 更新轨迹
            for m in matched:
                # !// track和det位置互换
                self.trackers[m[0]].update(dets[m[1], :])
                self.trackers[m[0]].update_emb(
                    dets_embs[m[1]], alpha=dets_alpha[m[1]])
                # self.trackers[m[1]].update(dets[m[0], :])
                # self.trackers[m[1]].update_emb(
                #     dets_embs[m[0]], alpha=dets_alpha[m[0]])
            
        else:
            matched, unmatched_trks, unmatched_dets = np.empty(
                (0, 2), dtype=int), np.empty((0, 5), dtype=int), np.arange(len(detection_indices))
        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            motion_pre_assign = self.motion_third_associate(
                dets, last_boxes, unmatched_dets, unmatched_trks, tracks_info,self.iou_threshold)
            # 外观预匹配
            left_dets_embs = dets_embs[unmatched_dets]
            left_trks_embs = [trk_embs[i] for i in unmatched_trks]
            left_dets = dets[unmatched_dets]
            left_trks = trks[unmatched_trks]
            # left_trks_embs = trk_embs[unmatched_trks]
            appearance_pre_assign, emb_cost = appearance_associate(
                left_dets_embs,
                left_trks_embs,
                left_dets, left_trks,
                unmatched_trks, unmatched_dets,
                tracks_info,
                self.gate, self.iou_threshold)
            # 双轮匹配
            matched, unmatched_trks, unmatched_dets = min_cost_matching(
                motion_pre_assign, appearance_pre_assign, self.alpha_gate)
            # print('matched: ', matched)
            # 更新轨迹
            for m in matched:
                self.trackers[int(m[0])].update(dets[int(m[1]), :])
                self.trackers[int(m[0])].update_emb(
                    dets_embs[int(m[1])], alpha=dets_alpha[int(m[1])])
        
        # for trk in reversed(self.trackers):
        #     print('last_observation2222222: ',trk.last_observation[:4])

        print('unmatch_track: ', unmatched_trks)
        print('unmatch_dets: ', unmatched_dets)
        for m in unmatched_trks:
            self.trackers[m].update(None)
        # print("second_len: ", len(self.trackers))

        # for trk in reversed(self.trackers):
        #     print('last_observation11111: ',trk.last_observation[:4])


        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i, :], delta_t=self.delta_t, emb=dets_embs[i], alpha=dets_alpha[i], new_kf=not self.new_kf_off
            )
            self.trackers.append(trk)
        # print("len_det2222: ", len(dets))
        i = len(self.trackers)
        # print('len_i11: ', i)

        # 测试track
        # for trk in reversed(self.trackers):
        #     print('track_state: ',trk.get_state()[0])
        #     # if trk.last_observation.sum() < 0:
        #     #     d = trk.get_state()[0]
        # for trk in reversed(self.trackers):
        #     print('last_observation: ',trk.last_observation[:4])

        # 提取当前帧跟踪的结果
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            # 当前跟踪到的 & (连续min_hits次被跟踪到 or 视频帧号<=min_hits)
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            # print('ret11: ', ret)
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        print("det_len333: ", len(ret))
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        self.cmc.dump_cache()
        self.embedder.dump_cache()

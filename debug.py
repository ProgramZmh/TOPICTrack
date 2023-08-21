import torch
import torch.nn.functional as F
import numpy as np


def get_track_feat(data):
    # 假设这里是获取轨迹或检测的特征向量的函数
    # 返回一个numpy数组作为示例
    return torch.from_numpy(np.random.rand(len(data), 128, 1))


def reconsdot_distance(tracks, detections, tmp=100):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float_)
    if cost_matrix.size == 0:
        return cost_matrix, None
    det_features_ = get_track_feat(detections)
    track_features_ = get_track_feat(tracks)

    det_features = F.normalize(det_features_, dim=1)
    track_features = F.normalize(track_features_, dim=1)

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

    recons_ftrk = recons_ftrk.permute(0, 2, 1).view(ntrk, nst*ndim, ndet)
    recons_ftrk_norm = F.normalize(recons_ftrk, dim=1)
    recons_fdet = recons_fdet.permute(0, 2, 1).view(ndet, nsd*ndim, ntrk)
    recons_fdet_norm = F.normalize(recons_fdet, dim=1)

    dot_td = torch.einsum('tad,ta->td', recons_ftrk_norm,
                          F.normalize(ftrk.reshape(ntrk, nst*ndim), dim=1))
    dot_dt = torch.einsum('dat,da->dt', recons_fdet_norm,
                          F.normalize(fdet.reshape(ndet, nsd*ndim), dim=1))

    cost_matrix = 1 - 0.5 * (dot_td + dot_dt.transpose(0, 1))
    cost_matrix = cost_matrix.detach().cpu().numpy()

    return cost_matrix


# 创建轨迹和检测数据
tracks = [1, 2, 3]  # 假设这里存储了轨迹数据
detections = [4, 5, 6]  # 假设这里存储了检测数据
cost = reconsdot_distance(tracks, detections, tmp=100)
print(cost)

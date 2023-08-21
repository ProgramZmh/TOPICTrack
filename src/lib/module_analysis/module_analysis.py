'''
Author: your name
Date: 2022-01-04 11:55:25
LastEditTime: 2022-01-04 15:03:39
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /ctracker/module_analysis.py
'''
import json


def load_gt(gt_file_path, seq_length):
    """加载gt的bbox，并按每一帧来划分bbox

    Args:
        gt_file_path ([type]): 路径
        seq_length ([type]): 序列frames

    Returns:
        [type]: [description]
    """
    with open(gt_file_path, 'r') as f:
        groundtruth_det = f.readlines()

    group_gt_det = {}
    det_line = 0
    for f_ in range(seq_length):
        cur_frame_dets = []
        start_line = groundtruth_det[det_line:]
        for det in start_line:
            frame = int(det.strip("\n").split(",")[0])

            if frame == f_ + 1:
                cur_frame_dets.append(det)
                det_line += 1
            else:
                break

        group_gt_det[f_+1] = []
        for det_str in cur_frame_dets:
            det = det_str.strip("\n").split(",")
            xmin = float(det[2])
            ymin = float(det[3])
            w = float(det[4])
            h = float(det[5])
            bbox = [xmin, ymin, w, h]

            group_gt_det[f_+1].append(bbox)

    return group_gt_det


def tlwh2xyxy(bbox):
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[0]+bbox[2], bbox[1]+bbox[3]

    return [x1, y1, x2, y2]


def save_det_res(results_file, det_results):
    with open(results_file, 'w') as f:
        f.write(json.dumps(det_results, indent=4))


def collect_det_res(det_results, frame, boxes, confs):
    for box, conf in zip(boxes, confs):
        x = float(box[0])
        y = float(box[1])
        w = float(box[2] - box[0] + 1)
        h = float(box[3] - box[1] + 1)
        det_results.append({'image_id': frame+1,
                            'category_id': 1,
                            'bbox': [x, y, w, h],
                            'score': float(conf)})

    return det_results

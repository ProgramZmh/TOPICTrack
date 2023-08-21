'''
Author: your name
Date: 2022-01-03 16:47:08
LastEditTime: 2022-01-04 15:36:57
LastEditors: your name
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /ctracker/module_analysis/mot_to_coco_format.py
'''
import json
import os


def mot2coco(DATA_PATH, sequence):
    seq_dir = os.path.join(DATA_PATH, sequence)

    # init coco format
    ret = {'images': [],
           'annotations': [],
           "categories": [{'name': "bee", 'id': 1}]}

    # annotations
    det_file_path = os.path.join(seq_dir, "det/det.txt")
    with open(det_file_path, "r") as f:
        all_dets = f.readlines()

    imgs = os.path.join(seq_dir, 'img1')
    det_line = 0
    for i, img_name in enumerate(sorted(os.listdir(imgs))):
        # images
        ret['images'].append({'file_name': os.path.join(imgs, img_name),
                              'id': i+1})

        # annotations
        cur_frame_dets = []
        start_line = all_dets[det_line:]
        for det in start_line:
            frame = str(det.strip("\n").split(",")[0])

            if frame in img_name:
                cur_frame_dets.append(det)
                det_line += 1
            else:
                break

        # generate coco annotations
        for det_str in cur_frame_dets:
            det = det_str.strip("\n").split(",")
            xmin = float(det[2])
            ymin = float(det[3])
            w = float(det[4])
            h = float(det[5])
            bbox = [xmin, ymin, w, h]
            area = round(bbox[2] * bbox[3], 2)

            ann = {'image_id': i+1,
                   'id': int(len(ret['annotations']) + 1),
                   'category_id': 1,
                   'bbox': bbox,
                   'iscrowd': 0,
                   'area': area}
            ret['annotations'].append(ann)

    # save
    out_path = os.path.join(seq_dir, 'annotations.json')
    json.dump(ret, open(out_path, 'w'))


if __name__ == '__main__':
    # dataset path
    DATA_PATH = '/home/caoxiaoyan/MOT_benchmark/BEE20/test/'
    sequence = "bee0009"

    mot2coco(DATA_PATH, sequence)

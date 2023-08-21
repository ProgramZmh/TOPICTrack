import os.path
import numpy as np
import cv2

def sort_output(txt_path):
    with open(txt_path, 'r') as f:
        list = []
        for line in f:
            list.append(line.strip())

    with open(txt_path, "w") as f:
        for item in sorted(list, key=lambda x: int(str(x).split(',')[0])):
            f.writelines(item)
            f.writelines('\n')
        f.close()


def draw_mot(video_id,gt_path):
    txt_name = gt_path + '/' + video_id + '.txt'  # txt文本内容
    file_path_img = 'data/bee/val/' + video_id + '/img1'  # img图片路径
    # 生成新的文件夹来存储画了bbox的图片
    if not os.path.exists('./pic_dets/' + video_id):
        os.mkdir('./pic_dets/' + video_id)
        print('The ./pic_dets/' + video_id + '  have create!')
    save_file_path = './pic_dets/' + video_id
    sort_output(txt_name)  # 这是一个对txt文本结果排序的代码，key=frame，根据帧数排序

    source_file = open(txt_name)
    # 把frame存入列表img_names
    img_names = []
    for line in source_file:
        staff = line.split(',')
        img_name = staff[0]
        img_names.append(img_name)

    # 将每个frame的bbox数目存入字典
    name_dict = {}
    for i in img_names:
        if img_names.count(i):
            name_dict[i] = img_names.count(i)
    # print(name_dict)
    source_file.close()

    source_file = open(txt_name)
    for idx in name_dict:
        # print(str(idx).rjust(6, '0'))
        # 因为图片名称是000001格式的，所以需要str(idx).rjust(6, '0')进行填充
        img = cv2.imread(os.path.join(file_path_img, str(idx).rjust(6, '0') + '.jpg'))
        for i in range(name_dict[idx]):
            line = source_file.readline()
            staff = line.split(',')
            id = staff[1]
            cls = staff[6].replace('\n','')
            box = staff[2:6]
            # print(id, box)
            # draw_bbox
            cv2.rectangle(img, (int(float(box[0])), int(float(box[1]))),
                          (int(float(box[0])) + int(float(box[2])), int(float(box[1])) + int(float(box[3]))),
                          (0, 255, 0), 2)
            # put_text
            cv2.putText(img, str(int(id)) + ' ' + str(float(cls)), (int(float(box[0])), int(float(box[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        # 保存图片
        cv2.imwrite(os.path.join(save_file_path, str(idx).rjust(6, '0') + '.jpg'), img)

    source_file.close()

def draw_det(video_id,gt_dir=None,pred_dir=None, save_dir=None):
    txt_name = os.path.join(gt_dir, video_id ,"gt",'gt.txt')  # txt文本内容
    pred_txt_name = os.path.join(pred_dir, video_id+'.txt')  # txt文本内容
    file_path_img = os.path.join(gt_dir, video_id ,'img1')  # img图片路径
    # 生成新的文件夹来存储画了bbox的图片
    save_file_path = save_dir + "/"+video_id
    if not os.path.exists(save_file_path ):
        os.mkdir(save_file_path )
        print('The ./pic_dets/' + video_id + '  have create!')
    sort_output(txt_name)  # 这是一个对txt文本结果排序的代码，key=frame，根据帧数排序
    sort_output(pred_txt_name)  # 这是一个对txt文本结果排序的代码，key=frame，根据帧数排序

    source_file = open(txt_name)
    pred_source_file = open(pred_txt_name)
    # 把frame存入列表img_names
    img_names = []
    for line in source_file:
        staff = line.split(',')
        img_name = staff[0]
        img_names.append(img_name)
    pred_img_names = []
    for line in pred_source_file:
        staff = line.split(',')
        img_name = staff[0]
        pred_img_names.append(img_name)

    # 将每个frame的bbox数目存入字典
    name_dict = {}
    for i in img_names:
        if img_names.count(i):
            name_dict[i] = img_names.count(i)
    pred_name_dict = {}
    for i in pred_img_names:
        if pred_img_names.count(i):
            pred_name_dict[i.zfill(6)] = pred_img_names.count(i)
    
    source_file.close()
    pred_source_file.close()

    source_file = open(txt_name)
    pred_source_file = open(pred_txt_name)
    for idx in name_dict:
        # print(str(idx).rjust(6, '0'))
        # 因为图片名称是000001格式的，所以需要str(idx).rjust(6, '0')进行填充
        img = cv2.imread(os.path.join(file_path_img, str(idx).rjust(6, '0') + '.jpg'))

        # 画gt
        for i in range(name_dict[idx]):
            line = source_file.readline()
            staff = line.split(',')
            id = staff[1]
            cls = staff[6].replace('\n','')
            box = staff[2:6]
            # print(id, box)
            # draw_bbox
            cv2.rectangle(img, (int(float(box[0])), int(float(box[1]))),
                          (int(float(box[0])) + int(float(box[2])), int(float(box[1])) + int(float(box[3]))),
                          (0, 255, 0), 2)
            # put_text
            cv2.putText(img, str(int(id)) + ' ' + str(""), (int(float(box[0])), int(float(box[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # 画pred
        for i in range(pred_name_dict[idx]):
            line = pred_source_file.readline()
            staff = line.split(',')
            id = staff[1]
            cls = staff[6].replace('\n','')
            box = staff[2:6]
            # print(id, box)
            # draw_bbox
            cv2.rectangle(img, (int(float(box[0])), int(float(box[1]))),
                          (int(float(box[0])) + int(float(box[2])), int(float(box[1])) + int(float(box[3]))),
                          (0, 0, 255), 2)
            # put_text
            cv2.putText(img, str(int(id)) + ' ' + str(int(float(cls))), (int(float(box[0])), int(float(box[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        # 保存图片
        cv2.imwrite(os.path.join(save_file_path, str(idx).rjust(6, '0') + '.jpg'), img)

    source_file.close()


if __name__ == '__main__':
    gt_dir = "data/bee16/val"
    pred_dir = "results/gmot_det"
    save_dir = "pic_dets/bee/bee_test"
    for name in os.listdir(gt_dir):
        print('The video ' + name.split('.')[0] + ' begin!')
        # draw_mot(name.split('.')[0],gt_path)
        draw_det(name,gt_dir,pred_dir,save_dir)
        print('The video ' + name.split('.')[0] + ' Done!')

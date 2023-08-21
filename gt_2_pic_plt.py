import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt
# 读取图像
image = Image.open('/remote-home/zhengyiyao/topictrack/data/bee/val/BEE2212/img1/000001.jpg')
# 创建画布并将图像加载进去
fig, ax = plt.subplots(1)
ax.imshow(image)

# with open('gt.txt', 'r') as file:
#        lines = file.readlines()
#        for line in lines:
#             # 解析一帧的目标信息
#             # frame_id, track_id, x, y, w, h = line.strip().split(',')
frame_id, track_id, x, y, w, h = 1, 2, 50, 50, 100, 100
y = image.size[1] - (int(y) + int(h))
# 创建跟踪框并设置样式
bbox = patches.Rectangle((int(x), int(y)), int(w), int(h), linewidth=2, edgecolor='yellow', facecolor='none')
ax.add_patch(bbox)
plt.text(int(x+2), int(y+2), track_id, color='r', fontsize=6, verticalalignment='top',bbox=dict(facecolor="yellow"))
plt.text(int(x+2), int(y+2), track_id, color='r', fontsize=6, verticalalignment='top',bbox=dict(facecolor="yellow"))
plt.savefig( 'exam_01.png')
# 展示图像
plt.show()


    
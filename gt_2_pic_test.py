import os.path
import numpy as np
import cv2
img = cv2.imread('/remote-home/zhengyiyao/topictrack/data/bee/val/BEE2212/img1/000001.jpg')
box = [50,50,50,50]
cv2.rectangle(img, (int(float(50)), int(float(50))),
                          (int(float(50)) + int(float(50)), int(float(50)) + int(float(50))),
                          (0,255,255), 2,cv2.LINE_4)
# put_text
cv2.rectangle(img, (50, 50),((50 + 20), (50 + 15)),(0,255,255), thickness=-1)
cv2.putText(img, str(2), (int(float(50)+5), int(float(50))+12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, bottomLeftOrigin=False)
cv2.putText(img, str(2.34), (int(float(50)), int(float(50)-5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, bottomLeftOrigin=False)

cv2.imwrite('test2.jpg', img)
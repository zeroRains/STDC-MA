import cv2
import numpy as np

path = f''
# img1 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/camvid/labels/0001TP_007260.png", -1)
# img2 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/camvid/labels/0001TP_007260.png", -1)
img1 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/camvid/labels/Seq05VD_f00300.png", -1)
img2 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/camvid/labels/Seq05VD_f00300.png", -1)
print(np.unique(img1))
img1[img1 == 9] = 255
img2[img2 == 10] = 255
cv2.imwrite('check.png', img1)
cv2.imwrite('check2.png', img2)


import cv2
import numpy as np

path = f''
# img1 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/camvid/labels/0001TP_007260.png", -1)
# img2 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/camvid/labels/0001TP_007260.png", -1)
img1 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/ADEChallengeData2016/annotations/training/ADE_train_00000344.png", -1)
img2 = cv2.imread("/home/disk2/ray/workspace/zerorains/stdc/data/ADEChallengeData2016/annotations/training/ADE_train_00000344.png", -1)
print(np.unique(img1))
img1[img1 == 150] = 255
img2[img2 == 0] = 255
cv2.imwrite('check.png', img1)
cv2.imwrite('check2.png', img2)


import torch
import numpy as np
import cv2


def edge_canny(image, thread1=60, thread2=150):
    # canny边缘检测
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return cv2.Canny(image, thread1, thread2)


def semantic_edge(img, class_num=8):
    """
    获取每一类的语意边缘
    input:
        img:预测结果或标签(b,h,w)
        classNum:类别数量
    output:
        img_edges:语义边缘
    """
    img = img.cpu().type(torch.LongTensor)
    pred_oht = torch.zeros([img.shape[0], class_num, img.shape[1], img.shape[2]])
    img = img.unsqueeze(1)
    img[img == 255] = 0
    pred_oht = pred_oht.scatter_(index=img, dim=1, value=255)
    pred_edges = []
    for j in range(class_num):
        temp_pred = []
        for i in range(img.shape[0]):
            pred_edge = edge_canny(pred_oht[i][j].numpy().astype(np.uint8))
            pred_edge[pred_edge == 255] = 1
            temp_pred.append(pred_edge)
        pred_edges.append(torch.from_numpy(np.array(temp_pred)))
    return pred_edges

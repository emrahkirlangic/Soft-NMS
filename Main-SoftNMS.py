# -*- coding:utf-8 -*-
# YOLOv8 + Soft-NMS + Renkli kutular (etiketsiz) + Görsel kaydı
# by Emrah & ChatGPT

import time
import numpy as np
import torch
import cv2
from ultralytics import YOLO

# SOFT-NMS ALGORİTMASI
def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]

    scores = box_scores.clone()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        yy1 = np.maximum(dets[i, 0].cpu().numpy(), dets[pos:, 0].cpu().numpy())
        xx1 = np.maximum(dets[i, 1].cpu().numpy(), dets[pos:, 1].cpu().numpy())
        yy2 = np.minimum(dets[i, 2].cpu().numpy(), dets[pos:, 2].cpu().numpy())
        xx2 = np.minimum(dets[i, 3].cpu().numpy(), dets[pos:, 3].cpu().numpy())

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    keep = dets[:, 4][scores > thresh].int()
    return keep

# ANA FONKSİYON
def main():
    model_path = "best.pt"
    image_path = "test.jpg"

    model = YOLO(model_path)
    img = cv2.imread(image_path)

    results = model.predict(source=img, verbose=False)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    labels = results[0].boxes.cls

    # NMS için format [y1, x1, y2, x2]
    boxes_swapped = boxes[:, [1, 0, 3, 2]]

    cuda = torch.cuda.is_available()
    if cuda:
        boxes_swapped = boxes_swapped.cuda()
        scores = scores.cuda()

    keep_indices = soft_nms_pytorch(boxes_swapped, scores, sigma=0.5, thresh=0.32, cuda=cuda)

    filtered_boxes = boxes[keep_indices]
    filtered_labels = labels[keep_indices]

    # 🖼️ Her nesne için farklı renkli kutu (etiket ve oran yok)
    for box, cls in zip(filtered_boxes, filtered_labels):
        x1, y1, x2, y2 = map(int, box)
        np.random.seed(int(cls.item()))
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 📸 KUTULU GÖRSELİ KAYDET
    cv2.imwrite("output.jpg", img)

    # GÖSTER (opsiyonel)
    cv2.imshow("Soft-NMS Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Çalıştır
if __name__ == "__main__":
    main()

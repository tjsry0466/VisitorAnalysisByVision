import cv2
import numpy as np
import torch
from yolov5.utils.general import scale_coords
from etc import bbox_rel

def deepsort_input(img, im0, det, deepsort):
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    bbox_xywh = []
    confs = []
    # Adapt detections to deep sort input format
    for *xyxy, conf, cls in det:
        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
        obj = [x_c, y_c, bbox_w, bbox_h]
        bbox_xywh.append(obj)
        confs.append([conf.item()])

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, im0)
    return outputs
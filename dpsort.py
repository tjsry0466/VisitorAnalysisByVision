import cv2
import numpy as np
import torch
from etc import bbox_rel

def deepsort_input(im0, bbox_xywh, confs, deepsort):
    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, im0)
    return outputs
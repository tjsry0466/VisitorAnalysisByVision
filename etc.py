import datetime
from yolov5.utils.general import scale_coords


def compute_color_for_labels(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color) # 사람마다 색깔 설정

def get_now():
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    return nowDatetime  # 2015-04-19 12:11:32\

def bbox_rel(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h #박스 영역(사람의 그어지는거)

def convert_tensor_xywh(img, im0, det):
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    bbox_xywh = []
    confs = []
    # Adapt detections to deep sort input format
    for *xyxy, conf, cls in det:
        # print(xyxy)
        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
        obj = [x_c, y_c, bbox_w, bbox_h]
        bbox_xywh.append(obj)
        confs.append([conf.item()])
    return (bbox_xywh, confs)
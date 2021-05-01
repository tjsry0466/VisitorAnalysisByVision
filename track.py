import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0, './yolov5')
import cv2
import torch
import torch.backends.cudnn as cudnn 

from yolov5.utils.datasets import LoadStreams
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

from settings import load_face_model, load_mask_model
from draws import draw_boxes, draw_face_and_mask_area
from detactions import detect_and_predict_mask
from dpsort import deepsort_input


def detect():
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
    device = select_device('')
    model = torch.load('yolov5/weights/yolov5s.pt', map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    model.half()  # to FP16 # tensorflow 모델값 저장
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams('0', img_size=640)

    faceNet = load_face_model()
    maskNet = load_mask_model()
    
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() 
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # object detaction
        pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred, 0.4, 0.5, classes=[0], agnostic=True)

        # face and mask detaction
        im0 = im0s[0].copy()
        (locs, preds) = detect_and_predict_mask(im0, faceNet, maskNet)

        # iterate object detection result
        for i, det in enumerate(pred):
            # if it has not result
            if det is None or not len(det):
                deepsort.increment_ages()
                continue
                
            # process deepsort input
            outputs = deepsort_input(img, im0,  det, deepsort)
            if len(outputs) < 1:
                continue
            # draw object boxes
            draw_boxes(im0, outputs)
        
        # draw face and mask detaction results
        for (box, pred) in zip(locs, preds):
            draw_face_and_mask_area(im0, box, pred)
        
        # display image
        cv2.imshow('frame', im0)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration
        

if __name__ == '__main__':
    with torch.no_grad():
        detect()
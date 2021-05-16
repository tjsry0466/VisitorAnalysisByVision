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
from processing import classify_face, s3_face_upload
from etc import convert_tensor_xywh

class models():
    def __init__():
        self.deepsort_config = "deep_sort_pytorch/configs/deep_sort.yaml"
        self.yolov5_weight = "yolov5/weights/yolov5s.pt"
        self.res10_caffe_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.res10_caffe_config = "models/deploy.prototxt.txt"
        self.mask_model = "models/mask_detector.model"

    def get_deepsort_model():
        self.cfg = get_config()
        self.cgf.merge_from_file(self.deepsort_config)
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                        max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
        return self.deepsort
    
    def get_yolov5_model():
        self.device = select_device('')
        self.model = torch.load(self.yolov5_weight, map_location=self.device)['model'].float()  # load to FP32
        self.model.to(self.device).eval()
        self.model.half()  # to FP16 # tensorflow 모델값 저장
        cudnn.benchmark = True  # set True to speed up constant image size inference
        return self.model

    def load_face_model():
        self.faceNet = cv2.dnn.readNetFromCaffe(self.res10_caffe_config, self.res10_caffe_model)
        return self.faceNet

    def load_mask_model():
        self.maskNet = load_model(self.mask_model)
        return self.maskNet
        

def detect():
    fdb = {}
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
        im0c = im0.copy()
        (locs, preds) = detect_and_predict_mask(im0, faceNet, maskNet)

        # iterate object detection result
        obj_ = pred[0]
        
        if len(obj_):
            bbox_xywh, confs = convert_tensor_xywh(img, im0,  obj_)
            outputs = deepsort_input(im0, bbox_xywh, confs, deepsort)
            if len(outputs) > 0:
                draw_boxes(im0, outputs)
        else:
            outputs = []
            deepsort.increment_ages()
        
        fdb = classify_face(frame_idx, im0, im0c, outputs, (locs, preds), fdb)
        # fdb = s3_face_upload(frame_idx, fdb)
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
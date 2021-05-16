import cv2
import torch

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from tensorflow.keras.models import load_model
from yolov5.utils.torch_utils import select_device


class Models():
    def __init__(self):
        self.deepsort_config = "deep_sort_pytorch/configs/deep_sort.yaml"
        self.yolov5_weight = "yolov5/weights/yolov5s.pt"
        self.res10_caffe_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.res10_caffe_config = "models/deploy.prototxt.txt"
        self.mask_model = "models/mask_detector.model"

    def get_deepsort_model(self):
        self.cfg = get_config()
        self.cfg.merge_from_file(self.deepsort_config)
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                        max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT, nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
        return self.deepsort
    
    def get_yolov5_model(self):
        self.device = select_device('')
        self.model = torch.load(self.yolov5_weight, map_location=self.device)['model'].float()  # load to FP32
        self.model.to(self.device).eval()
        self.model.half()  # to FP16 # tensorflow 모델값 저장
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        return self.model, self.device

    def get_face_model(self):
        self.faceNet = cv2.dnn.readNetFromCaffe(self.res10_caffe_config, self.res10_caffe_model)
        return self.faceNet

    def get_mask_model(self):
        self.maskNet = load_model(self.mask_model)
        return self.maskNet
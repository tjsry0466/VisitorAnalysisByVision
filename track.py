import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.insert(0, './yolov5')
import cv2
import torch

from yolov5.utils.datasets import LoadStreams
from yolov5.utils.general import non_max_suppression

from config import Models
from draws import draw_boxes, draw_face_and_mask_area
from detactions import detect_and_predict_mask
from dpsort import deepsort_input
from processing import classify_face, s3_face_upload
from etc import convert_tensor_xywh        

def detect():
    fdb = {}
    models = Models()
    deepsort = models.get_deepsort_model()
    model, device = models.get_yolov5_model()
    faceNet = models.get_face_model()
    maskNet = models.get_mask_model()

    dataset = LoadStreams('0', img_size=640)
    
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
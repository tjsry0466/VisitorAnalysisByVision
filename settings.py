import cv2
from tensorflow.keras.models import load_model

def load_face_model():
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt.txt"
    faceNet = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return faceNet

def load_mask_model():
    modelFile = 'models/mask_detector.model'
    maskNet = load_model(modelFile)
    return maskNet
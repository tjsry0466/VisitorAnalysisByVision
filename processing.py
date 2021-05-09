from etc import convert_tensor_xywh
from yolov5.utils.general import scale_coords

def classify_face(frame_idx, im0, obj_, face_):
    face_box, face_pred = face_
    print(f'---------------{frame_idx} process----------------')
    print('object:', end=' ')
    print(obj_)
    print('face:', end=' ')
    print(face_box)
    print(face_pred)
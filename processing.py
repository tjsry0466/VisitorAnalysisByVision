from etc import convert_tensor_xywh
from yolov5.utils.general import scale_coords
import numpy as np

def classify_face(frame_idx, im0, obj_, face_):
    face_box, face_pred = face_
    print(f'---------------{frame_idx} process----------------')
    obj_len = len(obj_)
    face_len = len(face_box)
    print(f'obj: {obj_len}, face: {face_len}')
    for i in range(face_len):
        for j in obj_:
            o_start_x, o_start_y, o_end_x, o_end_y, o_id = j
            f_start_x, f_start_y, f_end_x, f_end_y = face_box[i]
            mask, no_mask = face_pred[i]
            mask_label = True if mask > no_mask else False

            print(o_start_x, o_start_y, o_end_x, o_end_y, o_id)
            print(f_start_x, f_start_y, f_end_x, f_end_y, mask_label)
            # print(np.mean(o_start_x, o_end_x))
            

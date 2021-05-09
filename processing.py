from etc import convert_tensor_xywh
from yolov5.utils.general import scale_coords
import numpy as np

def classify_face(frame_idx, im0, im0c, obj_, face_, fdb):
    face_box, face_pred = face_
    print(f'---------------{frame_idx} process----------------')
    obj_len = len(obj_)
    face_len = len(face_box)
    print(f'obj: {obj_len}, face: {face_len}')
    # 10프레임마다 한번씩 계산
    for i in range(face_len):
        f_start_x, f_start_y, f_end_x, f_end_y = face_box[i]
        mask, no_mask = face_pred[i]
        mask_label = True if mask > no_mask else False

        # 얼굴 다 보이도록 15픽셀 만큼 늘려줌
        f_start_x-=15; f_end_x+=15; f_start_y-=15; f_end_y+=15
        if f_start_x < 0: f_start_x = 0
        if f_end_x > 640: f_end_x = 640
        if f_start_y < 0: f_start_y = 0
        if f_end_y > 640: f_end_y = 640

        f_w = f_end_x-f_start_x
        f_h = f_end_y-f_start_y

        print(f_start_x, f_start_y, f_end_x, f_end_y, mask_label)
        for j in obj_:
            o_start_x, o_start_y, o_end_x, o_end_y, o_id = j    
            f_mean_x, f_mean_y = (f_start_x+f_end_x)/2, (f_start_y+f_end_y)/2

            # 얼굴 위치 평균이 object안에 위치하는경우
            if o_start_x <= f_mean_x <= o_end_x and o_start_y <= f_mean_y <= o_end_y:
                if not o_id in fdb:
                    # 얼굴이 존재하지 않는 다면 저장
                    print(fdb)
                    fdb[o_id] = {
                        'f_xyxy': (f_start_x, f_start_y, f_end_x, f_end_y),
                        'f_arr': im0[f_start_y:f_end_y,f_start_x:f_end_x],
                        'f_mask': mask_label,
                        'f_wh' : (f_w, f_h),
                    } 
                else:
                    # 얼굴이 존재한다면 얼굴의 크기가 더 큰경우 저장
                    fdb_w, fdb_h = fdb[o_id]['f_wh']
                    if f_w*f_h > fdb_w*fdb_h:
                        fdb[o_id] = {
                            'f_xyxy': (f_start_x, f_start_y, f_end_x, f_end_y),
                            'f_arr': im0c[f_start_y:f_end_y,f_start_x:f_end_x],
                            'f_mask': mask_label,
                            'f_wh' : (f_w, f_h),
                        }
                print(o_id, ':', o_start_x, o_start_y, o_end_x, o_end_y)    
                print(f'face in object_id: {o_id}')
                f_shape = fdb[o_id]['f_arr'].shape
                # im0[0:f_shape[0],0:f_shape[1]] = fdb[o_id]['f_arr']
                    
            

import numpy as np
import cv2
import boto3
from multiprocessing import Process, Queue
from yolov5.utils.general import scale_coords
from etc import convert_tensor_xywh, get_uuid
s3 = boto3.client('s3')

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

        # print(f_start_x, f_start_y, f_end_x, f_end_y, mask_label)
        for j in obj_:
            o_start_x, o_start_y, o_end_x, o_end_y, o_id = j    
            f_mean_x, f_mean_y = (f_start_x+f_end_x)/2, (f_start_y+f_end_y)/2

            # 얼굴 위치 평균이 object안에 위치하는경우
            if o_start_x <= f_mean_x <= o_end_x and o_start_y <= f_mean_y <= o_end_y:
                if not o_id in fdb:
                    # 얼굴이 존재하지 않는 다면 저장
                    # print(fdb)
                    fdb[o_id] = {
                        'f_xyxy': (f_start_x, f_start_y, f_end_x, f_end_y),
                        'f_arr': im0c[f_start_y:f_end_y,f_start_x:f_end_x],
                        'f_mask': mask_label,
                        'f_wh' : (f_w, f_h),
                        'updatedFrame': frame_idx, 
                        'detectedFrame': frame_idx
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
                            'updatedFrame': frame_idx, 
                            'detectedFrame': frame_idx
                        }
                    else:
                        fdb[o_id]['detectedFrame'] = frame_idx
                # print(o_id, ':', o_start_x, o_start_y, o_end_x, o_end_y)    
                # print(f'face in object_id: {o_id}')
                f_shape = fdb[o_id]['f_arr'].shape
                im0[0:f_shape[0],0:f_shape[1]] = fdb[o_id]['f_arr']
                # cv2.imwrite('1.jpg', fdb[o_id]['f_arr'])
    return fdb

def upload_s3(face_img):
    # img = cv2.imread("1.jpg", cv2.IMREAD_COLOR) 
    uuid = get_uuid()
    data_serial = cv2.imencode('.jpg', face_img)[1].tobytes()
    s3.put_object(Bucket="facecog-bucket", Key =f'upload/visitor_{uuid}.jpg', Body=data_serial, ACL='public-read')

def s3_face_upload(frame_idx, fdb):
    delete_idx = []
    for key, value in fdb.items():
        # print(value['updatedFrame'])
        detectedFrame = value['detectedFrame']

        if frame_idx - detectedFrame >= 50:
            print('upload_s3', key)
            f_arg = value['f_arr'].copy()
            # th1 = Process(target=upload_s3, args=f_arg)
            # th1.start()
            # th1.join()
            upload_s3(f_arg)
            delete_idx.append(key)
            # th1.start()
            # th1.join()

    # 처리한 idx 삭제
    for i in delete_idx:
        print('detelte', i)
        del fdb[i]
    print(fdb.keys())
    return fdb


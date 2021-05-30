import numpy as np
import cv2
import boto3
from etc import get_uuid
s3 = boto3.client('s3')

class Process():
    '''
    
    '''
    def __init__(self):
        self.status = 0
        self.largest_obj_size = 0
        self.largest_obj_id = 0
        self.detacted_obj_frame = 0
        self.frame_idx = 0
        self.fdb = {}
        

    def classify_face_and_body(self, frame_idx, im0, im0c, obj_, face_):
        face_box, face_pred = face_
        print(f'---------------{frame_idx} process----------------')
        print(f'obj: {len(obj_)}, face: {len(face_box)}')
        # 10프레임마다 한번씩 계산
        for i in range(len(face_box)):
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
            f_h = f_end_y-f_start_y\

            # print(f_start_x, f_start_y, f_end_x, f_end_y, mask_label)
            for j in obj_:
                o_start_x, o_start_y, o_end_x, o_end_y, o_id = j    
                f_mean_x, f_mean_y = (f_start_x+f_end_x)/2, (f_start_y+f_end_y)/2

                # 얼굴 위치 평균이 object안에 위치하는경우
                if not o_start_x <= f_mean_x <= o_end_x and not o_start_y <= f_mean_y <= o_end_y:
                    continue
                if not o_id in self.fdb:
                    self.fdb[o_id] = {
                        'f_xyxy': (f_start_x, f_start_y, f_end_x, f_end_y),
                        'f_arr': im0c[f_start_y:f_end_y,f_start_x:f_end_x],
                        'f_mask': mask_label,
                        'f_wh' : (f_w, f_h),
                        'updatedFrame': frame_idx, 
                        'detectedFrame': frame_idx
                    } 
                else:
                    # 얼굴이 존재한다면 얼굴의 크기가 더 큰경우 저장
                    fdb_w, fdb_h = self.fdb[o_id]['f_wh']
                    if f_w*f_h > fdb_w*fdb_h:
                        self.fdb[o_id] = {
                            'f_xyxy': (f_start_x, f_start_y, f_end_x, f_end_y),
                            'f_arr': im0c[f_start_y:f_end_y,f_start_x:f_end_x],
                            'f_mask': mask_label,
                            'f_wh' : (f_w, f_h),
                            'updatedFrame': frame_idx, 
                            'detectedFrame': frame_idx
                        }
                    else:
                        self.fdb[o_id]['detectedFrame'] = frame_idx
                # print(o_id, ':', o_start_x, o_start_y, o_end_x, o_end_y)
                # print(f'face in object_id: {o_id}')
                f_shape = self.fdb[o_id]['f_arr'].shape
                im0[0:f_shape[0],0:f_shape[1]] = self.fdb[o_id]['f_arr']

    def upload_s3(self, face_img):
        # img = cv2.imread("1.jpg", cv2.IMREAD_COLOR) 
        uuid = get_uuid()
        data_serial = cv2.imencode('.jpg', face_img)[1].tobytes()
        s3.put_object(Bucket="facecog-bucket", Key =f'upload/visitor_{uuid}.jpg', Body=data_serial, ACL='public-read')

    def s3_face_upload(self, frame_idx):
        delete_idx = []
        for key, value in self.fdb.items():
            # print(value['updatedFrame'])
            detectedFrame = value['detectedFrame']

            if frame_idx - detectedFrame >= 50:
                print('upload_s3', key)
                f_arg = value['f_arr'].copy()
                self.upload_s3(f_arg)
                delete_idx.append(key)
        self.delete_fdb_by_list(delete_idx)
        return len(delete_idx)

    def delete_fdb_by_list(self, lst):
        for i in delete_idx:
            del self.fdb[i]
        return len(self.fdb), len(lst)

    def next(self):
        print(self.fdb.keys())
                

                    
                
            
            


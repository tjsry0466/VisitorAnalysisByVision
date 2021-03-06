import numpy as np
import cv2
import boto3
import win32com.client
import threading
from etc import get_uuid
from PIL import ImageFont, ImageDraw, Image

s3 = boto3.client('s3')

class Process():
    '''
    
    '''
    def __init__(self):
        self.message = ''
        self.recent_status = -1
        self.status = 0
        self.largest_obj_size = 0
        self.largest_obj_id = 0
        self.detacted_obj_frame = 0
        self.frame_idx = 0
        self.fdb = {}
        self.speak_flag = True
        self.speak_status = 0
        self.deleted_id = []
        

    def classify_face_and_body(self, frame_idx, im0, im0c, obj_, face_):
        self.frame_idx = frame_idx
        face_box, face_pred = face_
        print(f'---------------{frame_idx} process----------------')
        print(f'obj: {len(obj_)}, face: {len(face_box)}')
        # 10프레임마다 한번씩 계산
        for i in range(len(face_box)):
            f_start_x, f_start_y, f_end_x, f_end_y = face_box[i]
            mask, no_mask = face_pred[i]
            mask_label = True if mask > no_mask else False
            # print(mask_label)

            # 얼굴 다 보이도록 15픽셀 만큼 늘려줌
            offset_result = self.get_offset_area(f_start_x, f_end_x, f_start_y, f_end_y, 15)
            f_start_x, f_end_x, f_start_y, f_end_y = offset_result
            f_w = f_end_x-f_start_x
            f_h = f_end_y-f_start_y

            # print(f_start_x, f_start_y, f_end_x, f_end_y, mask_label)
            for j in obj_:
                o_start_x, o_start_y, o_end_x, o_end_y, o_id = j
                o_w = o_end_x-o_start_x
                o_h = o_end_y-o_start_y   
                f_mean_x, f_mean_y = (f_start_x+f_end_x)/2, (f_start_y+f_end_y)/2
                print(mask_label)
                # 얼굴 위치 평균이 object안에 위치하는경우
                if not o_start_x <= f_mean_x <= o_end_x and not o_start_y <= f_mean_y <= o_end_y:
                    continue
                if not o_id in self.fdb:
                    self.fdb[o_id] = {
                        'o_wh': (o_w,o_h),
                        'f_xyxy': (f_start_x, f_start_y, f_end_x, f_end_y),
                        'f_arr': im0c[f_start_y:f_end_y,f_start_x:f_end_x],
                        'f_mask': mask_label,
                        'f_wh' : (f_w, f_h),
                        'f_wh_now' : (f_w, f_h),
                        'updatedFrame': frame_idx, 
                        'detectedFrame': frame_idx
                    } 
                else:
                    # 얼굴이 존재한다면 얼굴의 크기가 더 큰경우 저장
                    fdb_w, fdb_h = self.fdb[o_id]['f_wh']
                    if f_w*f_h > fdb_w*fdb_h:
                        self.fdb[o_id] = {
                            'o_wh': (o_w,o_h),
                            'f_xyxy': (f_start_x, f_start_y, f_end_x, f_end_y),
                            'f_arr': im0c[f_start_y:f_end_y,f_start_x:f_end_x],
                            'f_mask': mask_label,
                            'f_wh' : (f_w, f_h),
                            'f_wh_now' : (f_w, f_h),
                            'updatedFrame': frame_idx, 
                            'detectedFrame': frame_idx
                        }
                    else:
                        self.fdb[o_id]['detectedFrame'] = frame_idx
                        self.fdb[o_id]['f_mask'] = mask_label
                        self.fdb[o_id]['f_wh_now'] = (f_w, f_h)
                
                # f_shape = self.fdb[o_id]['f_arr'].shape
                # im0[0:f_shape[0],0:f_shape[1]] = self.fdb[o_id]['f_arr']

    def get_offset_area(self, x1, x2, y1, y2, offset):
        start = 0
        end = 0

        x1-=15; x2+=15; y1-=15; y2+=15
        if x1 < 0: x1 = 0
        if x2 > 640: x2 = 640
        if y1 < 0: y1 = 0
        if y2 > 640: y2 = 640
        return x1, x2, y1, y2
        
    def upload_s3(self, face_img):
        # img = cv2.imread("1.jpg", cv2.IMREAD_COLOR) 
        uuid = get_uuid()
        data_serial = cv2.imencode('.jpg', face_img)[1].tobytes()
        s3.put_object(Bucket="facerecog-bucket", Key =f'upload/visitor_{uuid}.jpg', Body=data_serial, ACL='public-read')
        print('upload image')

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
        self.get_tracking_object_num()

        if not self.speak_flag:
            return
        if self.tracking_object_num in self.deleted_id:
            return 
        if self.tracking_object_num == -1:
            self.status = 0
            self.message = ''
            return
        if self.status == 0:
            self.status = 1
            self.message = '안녕하세요. 반갑습니다'
            return

        mask_status = self.get_mask_detect_status()
        over_face_size = self.is_over_face_size()
        # print('mask', mask_status)
        # print('isoverface', over_face_size)
        
        if not mask_status:
            self.status=2
            self.message = '마스크를 확인해 주시기 바랍니다'
            return

        if not over_face_size:
            self.status=3
            self.message = '온도측정을 위해 조금 더 \n가까이 와주시기 바랍니다.'
            return 

        self.message = '온도 측정을 시작합니다.'
        self.status=4
        temp, hand_wasing = self.get_temp_and_handwashing()

        if 35.5 <= temp <= 37.5:
            self.status = 5
            self.message = f'온도 측정 결과:{temp} \n정상 온도입니다.\n입장 해주시기 바랍니다.'
            self.deleted_id.append(self.tracking_object_num)
            self.status = 0
            self.upload_s3(self.fdb[self.tracking_object_num]['f_arr'])
        else:
            self.status=5
            self.message = f'온도 측정 결과:{temp} \n비정상 온도입니다.\n관리자에게 문의해 주시기 바랍니다.'
            self.deleted_id.append(self.tracking_object_num)
            self.status = 0
            self.upload_s3(self.fdb[self.tracking_object_num]['f_arr'])
    
    def print_and_speak_message(self, frame):
        b,g,r,a = 255,255,255,0
        fontpath = "fonts/NanumGothic.ttf"
        font = ImageFont.truetype(fontpath, 20)
        img_pil = Image.fromarray(frame)
        if self.message:
            draw = ImageDraw.Draw(img_pil)
            draw.text((200, 50),  self.message, font=font, fill=(b,g,r,a))
        
        if self.status == self.recent_status:
            return np.array(img_pil)

        if self.speak_flag and self.speak_status != self.status:
            t = threading.Thread(target=self.speak_message)
            t.daemon = True
            t.start()

        self.recent_status = self.status

        return np.array(img_pil)

    def speak_message(self):
        self.speak_flag = False
        tts = win32com.client.Dispatch("SAPI.SpVoice")
        tts.Speak(self.message)
        self.speak_flag = True
        self.speak_status = self.status

    def get_temp_and_handwashing(self):
        return 36.5, True
    
    def get_tracking_object_num(self):
        max_size_object_id = -1
        size = 0
        for key,value in self.fdb.items():
            if value['detectedFrame'] == self.frame_idx:
                o_w, o_h = value['o_wh']
                if o_w*o_h > size:
                    max_size_object_id = key
                    size = o_w*o_h 
        if not max_size_object_id in self.deleted_id:
            self.tracking_object_num = max_size_object_id
        else:
            self.tracking_object_num = -1
        print('tracking:', self.tracking_object_num)    

    def get_mask_detect_status(self):
        mask_status = -1
        if self.tracking_object_num in self.fdb:
            mask_status = self.fdb[self.tracking_object_num]['f_mask']
        return mask_status
    
    def is_over_face_size(self, face_size_condition = 55000):
        result = -1
        if self.tracking_object_num in self.fdb:
            face_size = self.fdb[self.tracking_object_num]['f_wh_now']
            f_w, f_h = face_size
            # 마스크 착용시 height가 줄어드는 문제가 있어 보정
            if self.get_mask_detect_status:
                f_h +=100
                
            face_size = f_w*f_h
            result = True if face_size > face_size_condition else False
        return result
                
        

            

                    
                
            
            


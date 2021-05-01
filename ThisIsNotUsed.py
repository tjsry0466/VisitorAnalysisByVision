def caculate_face(detection, box, xyxy):
    ox1, oy1, ox2, oy2 = xyxy
    ow = ox2-ox1
    oh = oy2-oy1
    fx1 = int(ox1+ow*box.xmin); fx2 = fx1 + int(ow*box.width)
    fy1 = int(oy1+oh*box.ymin); fy2 = fy1 + int(oh*box.height)

    faceKeyPoints = [mp_face_detection.FaceKeyPoint.NOSE_TIP,
                    mp_face_detection.FaceKeyPoint.NOSE_TIP,
                    mp_face_detection.FaceKeyPoint.LEFT_EYE,
                    mp_face_detection.FaceKeyPoint.RIGHT_EYE,
                    mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION,
                    mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION,
                    mp_face_detection.FaceKeyPoint.MOUTH_CENTER]

    faceKeyPoints = list(map(lambda point: mp_face_detection.get_key_point(detection, point), faceKeyPoints))
    return {'xy':(fx1, fy1, fx2,fy2), 'points':faceKeyPoints}

# fdb = detect_faces(im0, bbox_xyxy, identities, fdb) 
def detect_faces(im0, bbox_xyxy, identities, fdb):
    for (ox1, oy1, ox2, oy2), object_id in zip(bbox_xyxy, identities):
        im0_copy = im0.copy()
        ow = ox2-ox1
        oh = oy2-oy1
        oim = im0[oy1:oy2,ox1:ox2]
        oim_shape = oim.shape

        results = face_detection.process(cv2.cvtColor(oim, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return fdb
        detection = results.detections[0]
        box = detection.location_data.relative_bounding_box
        cf = caculate_face(detection, box, (ox1, oy1, ox2, oy2))
        xy = cf['xy']
        points = cf['points']

        # 얼굴 및 랜드마크 그리기
        cv2.rectangle(im0, (xy[0], xy[1], xy[2]-xy[0], xy[3]-xy[1]), (255,255,255), 3) 
        for point in points:
            cv2.circle(im0, (int(ox1+ow*point.x), int(oy1+oh*point.y)), 2, (255,0,0),2)  

        # 현재 db에 데이터가 없다면 추가
        if not object_id in fdb:
            fdb[object_id] = cf
        
        # 데이터가 있는경우에 기존에 있는 데이터와 비교
        if not 'cv' in fdb[object_id]:
            fdb[object_id]['cv'] = 1

        ''' 오차 계산
        two_eye_length = left_eye.x - right_eye.x
        left_ear_eye_length = left_ear_tragion.x - left_eye.x
        right_ear_eye_length = right_eye.x - right_ear_tragion.x
        current_value = fw + (two_eye_length*200)
        '''
        val = (xy[2]-xy[0])+(points[2].x-points[3].x)
        if fdb[object_id]['cv'] < val:
            fdb[object_id]['cv'] = val

            # print(xy)
            fim = im0_copy[xy[1]:xy[3], xy[0]:xy[2]].copy()
            fim_shape = fim.shape
            fdb[object_id]['opf'] = fim
        # 이미지 출력
        im0[0:fdb[object_id]['opf'].shape[0],0:fdb[object_id]['opf'].shape[1]] = fdb[object_id]['opf']
    return fdb
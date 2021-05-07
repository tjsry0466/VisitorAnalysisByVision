from etc import convert_tensor_xywh


def classify_face(img, im0, obj_box, face_info):
    face_pred, face_box = face_info
    print('---------------process----------------')
    print('object:', end=' ')
    for det in obj_box:
        if det is None or not len(det):
            continue
        bbox_xywh, confs = convert_tensor_xywh(img, im0,  det)
        print(bbox_xywh)
    print('face:', end=' ')
    print(face_box)
    print(face_pred)
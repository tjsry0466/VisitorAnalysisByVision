import cv2
from etc import compute_color_for_labels


def draw_boxes(img, outputs):
    bbox = outputs[:, :4]
    identities = outputs[:, -1]
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d} '.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3) 
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def draw_face_and_mask_area(frame, box, pred):
    # font = cv2.FONT_HERSHEY_SIMPLEX 
    (startX, startY, endX, endY) = box
    (mask, withoutMask) = pred

    label = "Mask" if mask > withoutMask else "No Mask"
    color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

    cv2.putText(frame, label, (startX, startY - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
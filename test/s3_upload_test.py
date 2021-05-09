import cv2
import boto3
s3 = boto3.client('s3')

img = cv2.imread("1.jpg", cv2.IMREAD_COLOR) 
data_serial = cv2.imencode('.jpg', img)[1].tobytes()
s3.put_object(Bucket="facecog-bucket", Key = 'upload/test1.jpg', Body=data_serial, ACL='public-read')
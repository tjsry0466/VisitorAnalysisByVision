# VisitorAnalysisByVision
## 소개
pre-trained model을 활용하여 방문자 추적에 필요한 몇가지 기술을 구현합니다.

- yolov5 + deepsort + pytorch를 이용한 object detaction & tracking 구현
- opencv dnn + res10&ssd cafemodel을 이용한 face detaction 구현
- mask detector model 을 이용한 mask detactor 구현

## 실행 방법
- python track.py

## 개발 내역
### 05.01
- object detaction&tracking 구현 확인
- face&mask detaction 구현 확인
- 소스 리팩토링 및 기능별 모듈로 함수 분리

### 05.09
- object내에서 face 영역 검출 및 저장(object별로 큰 얼굴만 fdb에 저장)
- 검출한 face 영역 s3에 업로드

## 추후 개발 내역
- 방문자 카운팅을 위한 object tracking의 id별 출입 확인
- 성별, 나이, 동일인물 확인 작업

## 참고자료
[Yolov5_DeepSort_Pytorch github](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

[Face-Mask-Detection github](https://github.com/chandrikadeb7/Face-Mask-Detection)
# VisitorAnalysisByVision
## 소개
pre-trained model을 활용하여 방문자 추적에 필요한 몇가지 기술을 구현합니다.

- yolov5 + deepsort + pytorch를 이용한 object detaction & tracking 구현
- opencv dnn + res10&ssd cafemodel을 이용한 face detaction 구현
- mask detector model 을 이용한 mask detactor 구현

## 실행 방법
- python track.py

## 개발 내역

### 기능 소개
- 얼굴, 객체 인식
- 얼굴인식 로직 수행
![방문자 입장 수행 로직](https://github.com/tjsry0466/VisitorAnalysisByVision/blob/main/examples/logic.png)

### 05.01
- object detaction&tracking 구현 확인
- face&mask detaction 구현 확인
- 소스 리팩토링 및 기능별 모듈로 함수 분리

### 05.09
- object내에서 face 영역 검출 및 저장(object별로 큰 얼굴만 fdb에 저장)
- 검출한 face 영역 s3에 업로드

### 05.09
- 리팩토링 작업
- 모델 클래스 추가, 설정파일 분리, predict파일 분리, track.py 리팩토링

### 05.17~30
- 프로세스 클래스 추가 및 리팩토링

### 06.06~06.20
- 프로세스 클래스 개발 진행
- 방문자 분석 로직 추가

## 참고자료
[Yolov5_DeepSort_Pytorch github](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

[Face-Mask-Detection github](https://github.com/chandrikadeb7/Face-Mask-Detection)
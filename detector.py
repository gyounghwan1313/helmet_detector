
### 라이브러리 import
import cv2, glob, os, dlib
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


### 얼굴, 헬멧 인식기 불러오기
detector = dlib.get_frontal_face_detector() # 얼굴 인식기
halmet_model = load_model("cnn_model.h5") # 헬멧 인식기

### relu함수 정의
def relu(x):
    if x>0:
        return x
    else:
        return 0


### 판독 사진 불러오기
test_img= cv2.imread("C:/Users/skybl/Downloads/images5.jpg") #헬멧착용사진
test_img= cv2.imread("C:/Users/skybl/Downloads/test_image_non2.jpg") #헬멧미착용사진

# 얼굴 탐지
faces=detector(test_img,1)

# 얼굴 있는지 없는지 확인
if len(faces) == 0:
    print('no faces!') #얼굴이 없을경우

else:
    print("detect faces!") #얼굴이 있을경우
    face = faces[0]

    # 얼굴과 머리만 추출
    test_img_face = test_img[relu(int(face.top()-80)):int(face.bottom()+20),relu(int(face.left()-20)):int(face.right()+30)]

    cv2.imshow("face_image", test_img_face) # 얼굴과 머리 사진

    test_img_face = Image.fromarray(test_img_face, "RGB")
    test_img_face.convert("L")
    test_img_face = test_img_face.resize((100, 100))
    test_img_face = np.array(test_img_face)

    # 헬멧착용 여부를 판단
    test_imgs = []
    test_imgs.append(test_img_face)
    test_imgs = np.array(test_imgs)
    halmet_model.predict(test_imgs)
    print(["미착용", "착용"][np.argmax(halmet_model.predict(test_imgs))])


# 원본 이미지를 출력
cv2.imshow("image",test_img)
cv2.waitKey(0)

### 라이브러리 import
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from PIL import Image, ImageOps



### 사진 불러오기

## 헬멜 미착용 사진
raw_img=[]
y_raw=[]
files = glob.glob("D:/analysis_picture/gray_raw"+"/*.jpg")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.resize((100, 100))
    img = img.convert("RGB")
    imgdata = np.array(img)
    raw_img.append(imgdata)
    y_raw.append([1,0])

## 헬멜 미착용 사진_ImageDataGenerator
files = glob.glob("D:/analysis_picture/gray_raw_rot"+"/*.jpg")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.resize((100, 100))
    img = img.convert("RGB")
    imgdata = np.array(img)
    raw_img.append(imgdata)
    y_raw.append([1,0])


## 헬멧 착용 사진
files = glob.glob("D:/analysis_picture/gray_hat"+"/*.jpg")
hat_img=[]
y_hat=[]
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.resize((100,100))
    img = img.convert("RGB")
    imgdata = np.array(img)
    hat_img.append(imgdata)
    y_hat.append([0,1])

## 헬멧 미착용 사진_ImageDataGenerator
files = glob.glob("D:/analysis_picture/gray_hat_rot"+"/*.jpg")
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.resize((100,100))
    img = img.convert("RGB")
    imgdata = np.array(img)
    hat_img.append(imgdata)
    y_hat.append([0,1])


raw_img=np.array(raw_img)
hat_img=np.array(hat_img)
y_raw=np.array(y_raw)
y_hat=np.array(y_hat)

raw_img.shape
hat_img.shape
y_raw.shape
y_hat.shape

x=np.concatenate([raw_img,hat_img])
x.shape

y=np.concatenate([y_raw,y_hat])
y.shape

### data split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=11,stratify=y)
x_train.shape



#모델 생성
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding="same",input_shape=(100,100,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding="same",activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dropout(.5))
model.add(Dense(2,activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="nadam",metrics=["accuracy"])

model.summary()
model.fit(x_train,y_train, batch_size=32,epochs=2,validation_data=(x_test,y_test))

# 모델 평가
score=model.evaluate(x_test,y_test)
print('loss : ',score[0])
print('accuracy : ',score[1])

# 모델 저장
model.save("cnn_model.h5")




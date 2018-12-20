
# Facial expression recognition using CNN in Tensorflow

1. 폴더안에 `fer2013.csv` & `shape_predictor_68_face_landmarks.dat` & `haarcascade_frontalface_default.xml`를 넣을 것.

2. 다음의 코드를 실행함으로써 csv파일을 이미지로 바꿈.

    ```
    python convert_fer2013_to_images_and_landmarks.py
    ```
3. 학습 시작

    ``` 
    python train.py --train=yes
    ```

4. 학습 시키고 평가하기

    ```
    python train.py --train=yes --evaluate=yes

    학습을 두번 진행하는 이유는 정확도를 높이기 위해 실제 학습 한번 거쳤을때의 accuracy가 60~70%인데 
    두번 거치면 99%까지 상승하는 효과!     
    ```
<img width="465" alt="2018-12-05 2 42 41" src="https://user-images.githubusercontent.com/43809168/49492563-1d99b780-f89c-11e8-8c85-09e492788163.png">
놀라워라~

5. 입력한 이미지로 Predict 하는법

    ```
    python start.py

    이때 start.py의 starting('*.jpg')의 매개변수로 평가할 파일명을 넣어야함.
    서버쪽에서 안드로이드 스튜디오에서 받아온 파일 이름 매개변수로 사용할 것.
    ```

# 원 코드 출처

https://github.com/amineHorseman/facial-expression-recognition-using-cnn

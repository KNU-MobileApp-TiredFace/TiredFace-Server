
# Facial expression recognition using CNN in Tensorflow

1. 폴더안에 `fer2013.csv` 과 `shape_predictor_68_face_landmarks.dat` 를 넣을 것.

2. 다음의 코드를 실행함으로써 csv파일을 이미지로 바꿀 것.

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
```

5. 입력한 이미지로 Predict 하는법

```
python predict.py --image path/to/image.jpg
```
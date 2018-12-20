"""
@AmineHorseman
Sep, 12th, 2016
"""
import tensorflow as tf
from tflearn import DNN
import time
import numpy as np
import argparse
import dlib
import cv2
import os
from skimage.feature import hog
import matplotlib.pyplot as plt

from parameters import DATASET, TRAINING, NETWORK, VIDEO_PREDICTOR
from model import build_model

window_size = 24
window_step = 6


def load_model():
    model = None
    with tf.Graph().as_default():
        print("loading pretrained model...")
        network = build_model()
        model = DNN(network)
        if os.path.isfile(TRAINING.save_model_path):
            model.load(TRAINING.save_model_path)
        else:
            print("Error: file '{}' not found".format(TRAINING.save_model_path))
    return model


def get_landmarks(image, rects, predictor):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, NETWORK.input_size, window_step):
        for x in range(0, NETWORK.input_size, window_step):
            window = image[y:y+window_size, x:x+window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualise=False))
    return hog_windows


def predict(image, model, shape_predictor=None):
    # get landmarks
    if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks or NETWORK.use_hog_sliding_window_and_landmarks:
        face_rects = [dlib.rectangle(
            left=0, top=0, right=NETWORK.input_size, bottom=NETWORK.input_size)]
        face_landmarks = np.array(
            [get_landmarks(image, face_rects, shape_predictor)])
        features = face_landmarks
        if NETWORK.use_hog_sliding_window_and_landmarks:
            hog_features = sliding_hog_windows(image)
            hog_features = np.asarray(hog_features)
            face_landmarks = face_landmarks.flatten()
            features = np.concatenate((face_landmarks, hog_features))
        else:
            hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                  cells_per_block=(1, 1), visualise=True)
            hog_features = np.asarray(hog_features)
            face_landmarks = face_landmarks.flatten()
            features = np.concatenate((face_landmarks, hog_features))
        tensor_image = image.reshape(
            [-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict(
            [tensor_image, features.reshape((1, -1))])
        return get_emotion(predicted_label[0])
    else:
        tensor_image = image.reshape(
            [-1, NETWORK.input_size, NETWORK.input_size, 1])
        predicted_label = model.predict(tensor_image)
        return get_emotion(predicted_label[0])
    return None


def get_emotion(label):
    if VIDEO_PREDICTOR.print_emotions:
        print("- Angry: {0:.1f}%\n- Happy: {1:.1f}%\n- Sad: {2:.1f}%\n- Surprise: {3:.1f}%\n- Neutral: {4:.1f}%".format(
            label[0]*100, label[1]*100, label[2]*100, label[3]*100, label[4]*100))
    label = label.tolist()
    
    for i, v in enumerate(label):
        print(VIDEO_PREDICTOR.emotions[i])
        print(label[i])
    return VIDEO_PREDICTOR.emotions[4], label[4]


def starting(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    cv2.imwrite('new_image.jpg', gray)
    model = load_model()
    image = cv2.imread('new_image.jpg', 0)
    shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
    start_time = time.time()
    emotion, confidence = predict(image, model, shape_predictor)
    total_time = time.time() - start_time
    print("Prediction: {0} (confidence: {1:.1f}%)".format(emotion, confidence*100))
    print("time: {0:.1f} sec".format(total_time))
    return cropped,emotion,confidence*100





# parse arg to see if we need to launch training now or not yet
# parser = argparse.ArgumentParser()
# parser.add_argument("-i", "--image", help="Image file to predict")
# args = parser.parse_args()
# if args.image:
#     if os.path.isfile(args.image):
#         model = load_model()
#         image = cv2.imread(args.image, 0)
#         shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)
#         start_time = time.time()
#         emotion, confidence = predict(image, model, shape_predictor)
#         total_time = time.time() - start_time
#         print("Prediction: {0} (confidence: {1:.1f}%)".format(
#             emotion, confidence*100))
#         print("time: {0:.1f} sec".format(total_time))
#     else:
#         print("Error: file '{}' not found".format(args.image))

import math
from mimetypes import init
import random
from tkinter import *
from tkinter import ttk 
import pyttsx3
import threading
import speech_recognition as sr
import pyautogui
from PIL import Image
import imutils 
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.spatial import distance
import cv2
import PIL.ImageOps  
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from gtts import gTTS
from playsound import playsound
from timeit import default_timer as timer
from pygame import mixer
import time
import joblib

def create_ML_Model():
    catagories = os.listdir('image_dataset/')
   # print(catagories)
    path, dirs, files = next(os.walk("image_dataset/"))
    #print(path)

    data=[]
    for i in catagories:
        path1=os.path.join(path,i)
        label = catagories.index(i)
        for img in os.listdir(path1):
            imgpath = os.path.join(path1,img)
           # print(imgpath)
            img = cv2.imread(imgpath,0)
            try:
                img = cv2.resize(img,(50,50))
                image = np.array(img).flatten()
                data.append([image,label])
            except Exception as e:
                pass

    random.shuffle(data)
    features = []
    labels = []
    for feature,label in data:
        features.append(feature)
        labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state = 101)

    model = SVC(C=1, kernel='poly',gamma='auto',probability=True)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    acccuracy = model.score(X_test, y_test)
    print("prediction :",prediction)
    print("Accuracy :",acccuracy)

    filename = 'model.h5'
    joblib.dump(model, filename)

    model = joblib.load('model.h5')
    prediction = model.predict(X_test)
    acccuracy = model.score(X_test, y_test)
    print("prediction :",prediction)
    print("Accuracy :",acccuracy)
    

create_ML_Model()
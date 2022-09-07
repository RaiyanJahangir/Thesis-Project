from tkinter import *
from tkinter import ttk 
from subprocess import Popen
import pyautogui
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
from scipy.spatial import distance
import sys
import cv2
from PIL import Image
import PIL.ImageOps  
from PIL import ImageFilter
import numpy as np 
import pandas as pd 
import os
from os import listdir
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import imutils 
import random
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle #shuffling the data improves the model

model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2" # Link to model weights 
IMAGE_SHAPE = (224, 224)

layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,)) # Loading model weights
model = tf.keras.Sequential([layer]) # Building model object using loaded weights

currentLetter='A'
dotSize = 2
threshold = 20 
letterri = []
index = 0
root = Tk()
root.title("Test")
root.geometry("970x700")


def crop_contour(image, plot=False):
    intact=image
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray,5)
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.threshold(gray, 46, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    
    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    image = new_image
    
    return image


def gotoPractice():
    Popen(['python', 'practice.py'])
    root.withdraw()
    

def gotoTest():
    print("yo wassup")
    
def gotoCreate():
    Popen(['python', 'create.py'])
    root.withdraw() 

def getData(event):
    global currentLetter
    newLetter = letterChoice.get()
    if newLetter != currentLetter:
        currentLetter = newLetter
        wn.delete('all')
        print(currentLetter)

        
def clearCanvas():
    wn.delete('paint')
    
def evaluate():
    print("coming soon!!!")
    x, y = background.winfo_rootx()+122, background.winfo_rooty()
    w, h = background.winfo_width()-122, background.winfo_height()
    pyautogui.screenshot('screenshot.jpg', region=(x, y, w, h))
    image = Image.open('screenshot.jpg')
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('screenshot.jpg')

    ex_crop_img = crop_contour( cv2.imread('screenshot.jpg'), True)
    cv2.imwrite('screenshot.jpg',ex_crop_img)
    
    img_1='screenshot.jpg'
    img_2='images/'+currentLetter+'.jpg'

    # Calculations for image 1
    img_1 = Image.open(img_1).convert('L').resize(IMAGE_SHAPE) # Resizing the image to required size
    img_1 = np.stack((img_1,)*3, axis=-1) # Converting the image into a color representation for each pixel
    img_1 = np.array(img_1)/255.0 # Normalizing the values between 0 and 1
    embedding_img1 = model.predict(img_1[np.newaxis, ...]) # Extracting the features
    embedding_img1_np = np.array(embedding_img1) # Converting to numpy array
    flattened_feature_img1 = embedding_img1_np.flatten() # Converting matrix to a vector

    # Calculations for image 2
    img_2 = Image.open(img_2).convert('L').resize(IMAGE_SHAPE) # Resizing the image to required size
    img_2 = np.stack((img_2,)*3, axis=-1) # Converting the image into a color representation for each pixel
    img_2 = np.array(img_2)/255.0 # Normalizing the values between 0 and 1
    embedding_img2 = model.predict(img_2[np.newaxis, ...]) # Extracting the features
    embedding_img2_np = np.array(embedding_img2) # Converting to numpy array
    flattened_feature_img2 = embedding_img2_np.flatten() # Converting matrix to a vector

    methods = ['sqeuclidean', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
        'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
        'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
        'sokalmichener', 'sokalsneath', 'braycurtis', 'yule']

    mins = [0,0,0,0,0,0,0,0,0,0,0,0.475,0,0,0,0.475,0,0,0,0]
    maxes = [150,450, 2.5, 200, .5, .4, .4, 13, .6, 1, 0.5, 1, .6, 13, .4, .8, .5, .7, .5, .3]
    scores = []

    for i, m in enumerate(methods):

        metric = m # Try using any one of the methods listed above if needed
        dist_boyboy = distance.cdist([flattened_feature_img1], [flattened_feature_img2],metric)[0]      # Finding similarity 

        score = max(0, maxes[i] - dist_boyboy[0])
        #print(score)
        interval = maxes[i] - mins[i]
        score = score / interval * 100
        scores.append(score)

    print('Similarity is ',max(scores))


def paint(event):
    
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    color = "black"
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')

    
background=Canvas(root, width=970, height=700, bg='#eeeeee')
wn=Canvas(root, width=850, height=700, bg='white')

background.create_window(60,  50,window=Button(root,text='Practice', command=gotoPractice, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 190,window=Button(root,text='Test', command=gotoTest, bg='#eeeeee', fg='black', font=('helvetica', 15, 'bold')))
background.create_window(60, 120,window=Button(root,text='Create', command=gotoCreate, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 260,window=Button(root,text='clear', command=clearCanvas, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
strrng = StringVar() 
label = Label(background, text = "Select the Letter :", font = ("Times New Roman", 10))
label.configure(background='#eeeeee')
label.place(x=5,y=290)
letterChoice = ttk.Combobox(background, width = 10, textvariable = strrng) 



lines = []
with open(r'letters/letterlist.txt') as f:
    for row in f:
        lines.append(row.rstrip('\n'))
letterChoice['values'] = lines
    
letterChoice.bind("<<ComboboxSelected>>",getData)
letterChoice.current(0)
letterChoice.place(x=5, y=330)
wn.bind('<B1-Motion>', paint)
wn.place(x=120, y=0)
background.create_window(60, 400,window=Button(root,text='Evaluate', command=evaluate, bg='brown', fg='white', font=('helvetica', 15, 'bold')))

background.pack()



root.mainloop()

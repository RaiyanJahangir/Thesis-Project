from ctypes import alignment
import math
from mimetypes import init
from tkinter import *
from tkinter import ttk 
import threading
from cgitb import text
from email.mime import audio
import speech_recognition as sr
import pyttsx3

#test_imports
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



# main.py 
currentLetter='A'
dotSize = 5
threshold = 15
index = 0
strokeIndex = 1
strokePointIndex = 0
root = Tk()
root.title("Home")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
geometry = str(screen_width)
geometry = geometry+"x"
geometry = geometry+str(screen_height)
root.geometry(geometry)
session = 'Home'



# practice.py
practice_currentLetter='A'
practice_dotSize = 5
practice_threshold = 15
practice_index = 0
practice_strokeIndex = 1
practice_strokePointIndex = 0
practice_letter = []
practice_xshift = 0
practice_yshift = 0
practice_CurrentSizeD = 1
practice_dataset=[]
practice_started = False
practice_initX = 0
practice_initY = 0
practice_nStepAgo = [0,0]
practice_stepsCrossed = 0
practice_angleThresh = 20
say=False
command='Let\'s Start'
practice_letterChoiceV='A'
practice_fontsizeV = 1

def practice_makeDataset():
    global practice_dataset
    global practice_letter
    global practice_currentLetter
    global practice_CurrentSizeD
    global practice_index,practice_strokePointIndex,practice_strokeIndex
    global practice_started
    global practice_threshold
    global practice_dotSize
    wn.delete('all')
    practice_index = 0
    practice_strokePointIndex = 0
    practice_strokeIndex = 1
    practice_started = False
    practice_letter=[]
    practice_dataset = []
    filename = "letters/"+practice_currentLetter+".txt"
    f = open(filename, "r")
    lstr = f.readlines()
    for line in lstr:
        numberlist = line.split()
        # print(numberlist)
        pointlist = []
        for idx in range(0, len(numberlist),2):
            curpoint = [int(numberlist[idx]),int(numberlist[idx+1])]
            # print(curpoint)
            pointlist.append(curpoint)
        practice_dataset.append(pointlist)
        
    for stroke in practice_dataset:
        for point in stroke:
            point[0] = point[0]/practice_CurrentSizeD
            point[1] = point[1]/practice_CurrentSizeD
            
        
    practice_letter = practice_dataset[0]
    print("current practice_letter = ",practice_currentLetter)
    print("current size = ",practice_CurrentSizeD)
    
    practice_threshold = practice_threshold/practice_CurrentSizeD
    practice_dotSize = practice_dotSize/practice_CurrentSizeD
      

def practice_correctionMSGpoint(x,y,tx,ty):
    x1,y1,ovalStartx1,ovalStarty1 = x,y,tx,ty
    x2=x1+practice_threshold
    y2=y1+practice_threshold
    ovalEndx1,ovalEndy1 = tx,ty
    text = ""
    global command
    global say
    if x1 < ovalStartx1 and y1 < ovalStarty1:
        text = "Go Down Right"
    elif x1 > ovalEndx1 and y1 < ovalStarty1:
        text = "Go Down Left"
    elif x1 < ovalStartx1 and y1 > ovalEndy1:
        text = "Go Up Right"
    elif x1 > ovalEndx1 and y1 > ovalEndy1:
        text = "Go Up Left"
    elif y1 < ovalStarty1 and x1>=ovalStartx1 and x2<ovalEndx1:
        text = "Go Down"
    elif y1 > ovalEndy1 and x1>=ovalStartx1 and x2<ovalEndx1:
        text = "Go Up"
    elif x1 < ovalStartx1 and y1>=ovalStarty1 and y2<ovalEndy1:
        text = "Go Right"
    elif x1 > ovalEndx1 and y1>=ovalStarty1 and y2<ovalEndy1:
        text = "Go Left"
    
    command=text
    say=True
    return "Out of Line!!! "+text



def practice_responseNext(x,y,tx,ty):
    x1,y1,ovalStartx1,ovalStarty1 = x,y,tx,ty
    ovalEndx1,ovalEndy1 = tx,ty
    x2=x1+practice_threshold
    y2=y1+practice_threshold
    text = ""
    global command
    global say
    if x1 < ovalStartx1 and y1 < ovalStarty1:
        text = "Go Down Right"
    elif x1 > ovalEndx1 and y1 < ovalStarty1:
        text = "Go Down Left"
    elif x1 < ovalStartx1 and y1 > ovalEndy1:
        text = "Go Up Right"
    elif x1 > ovalEndx1 and y1 > ovalEndy1:
        text = "Go Up Left"
    elif y1 < ovalStarty1 and x1>=ovalStartx1 and x2<ovalEndx1:
        text = "Go Down"
    elif y1 > ovalEndy1 and x1>=ovalStartx1 and x2<ovalEndx1:
        text = "Go Up"
    elif x1 < ovalStartx1 and y1>=ovalStarty1 and y2<ovalEndy1:
        text = "Go Right"
    elif x1 > ovalEndx1 and y1>=ovalStarty1 and y2<ovalEndy1:
        text = "Go Left"
    
    command=text
    say=True
    return "keep on going!!! "+text


def practice_getData(event):
    global letterChoice
    global practice_currentLetter
    
    
    newLetter = letterChoice.get()
    if newLetter != practice_currentLetter:
        practice_currentLetter = newLetter
        practice_makeDataset()
        
def practice_getSize(event):
    global practice_dataset
    global fontsizeC
    global practice_CurrentSizeD
    data = fontsizeC.current()
    divisors = [4,2,(4/3),1,0.8,(2/3),(4/7)]
    newSizeD = divisors[data]
        
    if newSizeD != practice_CurrentSizeD:
        practice_CurrentSizeD = newSizeD
        practice_makeDataset()

#test.py
test_currentLetter='A'
test_dotSize = 2
test_threshold = 20 
test_letterri = []
test_index = 0
root = Tk()
root.title("Test")
root.geometry("1000x1000")

def test_crop_contour(image, plot=False):
    intact=image
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray,5)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
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


def test_getData(event):
    global test_currentLetter
    test_newLetter = letterChoice.get()
    if test_newLetter != test_currentLetter:
        test_currentLetter = test_newLetter
        wn.delete('all')
        print(test_currentLetter)

def test_evaluate():
    print("coming soon!!!")
    x, y = background.winfo_rootx()+122, background.winfo_rooty()
    w, h = background.winfo_width()-122, background.winfo_height()
    pyautogui.screenshot('screenshot.jpg', region=(x, y, w, h))
    image = Image.open('screenshot.jpg')
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('screenshot.jpg')

    ex_crop_img = test_crop_contour( cv2.imread('screenshot.jpg'), True)
    cv2.imwrite('screenshot.jpg',ex_crop_img)
    
    img_1='screenshot.jpg'
    img_2='images/'+test_currentLetter+'.jpg'

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
    
    if session == 'Home':
        paintH(event)
    elif session == 'Practice':
        paintP(event)
    elif session == 'Test':
        paintT(event)
    elif session == 'Create':
        paintC(event)

def paintH(event):
    
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    color = "red"
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')

def paintP(event):
    
    global practice_strokeIndex,practice_strokePointIndex,practice_index
    global practice_initX,practice_initY,practice_nStepAgo
    global practice_started
    global practice_dataset
    global command
    global say
    if practice_index == len(practice_letter):
        command='Done'
        say=True
        print("Done!!!")
        return 
    
    x = event.x 
    y = event.y 
    x1, y1 = (x-practice_dotSize), (y-practice_dotSize)
    x2, y2 = (x+practice_dotSize), (y+practice_dotSize)
    color = "black"
    
    
    targetX = 0
    targetY = 0
    if practice_started == False:
        practice_initX = x-practice_dataset[0][practice_index][0]
        practice_initY = y-practice_dataset[0][practice_index][1]
        practice_started = True
        for point in practice_letter:
            xc = point[0]+practice_initX
            yc = point[1]+practice_initY
            if xc>820 :
                print("too much on right side, go left")
                practice_makeDataset()
                break
            if xc<10 :
                print("too much on left side, go right")
                practice_makeDataset()
                break
            if yc<10 :
                print("too much on upperside, go down")
                practice_makeDataset()
                break
            if yc>690 :
                print("too much on bottom side, go up")
                practice_makeDataset()
                break
    
    targetX = practice_dataset[0][practice_index][0]+practice_initX
    targetY = practice_dataset[0][practice_index][1]+practice_initY
    angleNeed = 0
    draw_point = True
    
    
    
    if x>=targetX-practice_threshold and x<= targetX+practice_threshold and y>=targetY-practice_threshold and y<= targetY+practice_threshold:
        xt1=targetX-practice_threshold
        xt2=targetX+practice_threshold
        yt1=targetY-practice_threshold
        yt2=targetY+practice_threshold
        print("found!!!")
        wn.create_oval(xt1, yt1, xt2, yt2, fill='white', outline=color,tags='follow')
        practice_index=practice_index+1
        practice_strokePointIndex = practice_strokePointIndex+1
        if practice_index<len(practice_letter):
            targetX = practice_dataset[0][practice_index][0]+practice_initX
            targetY = practice_dataset[0][practice_index][1]+practice_initY
    
    else:
        pretX,pretY = 0,0
        if practice_index > 0:
            pretX = practice_dataset[0][practice_index-1][0]+practice_initX
            pretY = practice_dataset[0][practice_index-1][1]+practice_initY
            
        if not(x>=pretX-practice_threshold and x<= pretX+practice_threshold and y>=pretY-practice_threshold and y<= pretY+practice_threshold):
            print(practice_correctionMSGpoint(x,y,pretX,pretY))
            draw_point = False
    
    
    if draw_point:
        wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')
        print(practice_responseNext(x,y,targetX,targetY))
        


def paintC(event):
    
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    color = "pink"
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')

def paintT(event):
    
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    color = "black"
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')





def gotoHome():
    global session
    session = 'Home'

def gotoPractice():
    global session
    strrng = StringVar() 
    label = Label(background, text = "Select the Letter :", font = ("Times New Roman", 12))
    label.configure(background='#eeeeee')
    label.place(x=5,y=295)
    letterChoice = ttk.Combobox(background, width = 15, textvariable = strrng) 

    lines = []
    with open(r'letters/letterlist.txt') as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    letterChoice['values'] = lines
        
    letterChoice.bind("<<ComboboxSelected>>",practice_getData)
    letterChoice.current(0)
    letterChoice.place(x=5, y=320)

    strrng = StringVar() 
    label = Label(background, text = "Select Font size :", font = ("Times New Roman", 12))
    label.configure(background='#eeeeee')
    label.place(x=5,y=350)
    fontsizeC = ttk.Combobox(background, width = 15, textvariable = strrng) 

    background.create_window(60, 430,window=Button(root,text='Exit', command=root.destroy, bg='brown', fg='white', font=('helvetica', 15, 'bold')))

    lines = ['0(25%)','1(50%)','2(75%)','3(100%)','4(125%)','5(150%)','6(175%)']
    fontsizeC['values'] = lines
        
    fontsizeC.bind("<<ComboboxSelected>>",practice_getSize)
    fontsizeC.current(3)
    fontsizeC.place(x=5, y=375)

    practice_makeDataset()
    session = 'Practice'


def gotoTest():
    global session
    session = 'Test'
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
        
    letterChoice.bind("<<ComboboxSelected>>",test_getData)
    letterChoice.current(0)
    letterChoice.place(x=5, y=330)


def gotoCreate():
    global session
    session = 'Create'
def clear():
    pass

background=Canvas(root, width=screen_width, height=screen_height, bg='#eeeeee')

background.create_window(60,  50,height = 40,width = 100,window=Button(root,text='Home', command=gotoHome, bg='Brown', fg='white', font=('helvetica', 12, 'bold')))
background.create_window(60, 120,height = 40,width = 100,window=Button(root,text='Practice', command=gotoPractice, bg='Brown', fg='white', font=('helvetica', 12, 'bold')))
background.create_window(60, 190,height = 40,width = 100,window=Button(root,text='Test', command=gotoTest, bg='brown', fg='white', font=('helvetica', 12, 'bold')))
background.create_window(60, 260,height = 40,width = 100,window=Button(root,text='Create', command=gotoCreate, bg='brown', fg='white', font=('helvetica', 12, 'bold')))
background.create_window(60, 330,height = 40,width = 100,window=Button(root,text='clear', command=None, bg='brown', fg='white', font=('helvetica', 12, 'bold')))

wn=Canvas(root, width=screen_width, height=screen_height, bg='white')
wn.bind('<B1-Motion>', paint)


wn.place(x=120, y=0)



background.pack()

root.mainloop()




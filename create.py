from tkinter import *
from tkinter import ttk 
from subprocess import Popen
import math
import pyautogui
from PIL import Image
import PIL.ImageOps  
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


currentLetter='A'
dotSize = 2
threshold = 20 
letterri = []
index = 0
root = Tk()
root.title("Create")
root.geometry("970x700")
letter = []
making = ""
maxX = -1
maxY = -1
minX = 10000
minY = 10000

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
    Popen(['python', 'test.py'])
    root.withdraw()
    
def gotoCreate():
    print("yo wassup") 

def getLetter():
    global making
    newLetter = txtfld.get()
    making = newLetter.strip()
    if making[0] <= 'Z' and making[0] >= 'A':
        making = "cap"+making
    else:
        making = "SML"+making

    print(making)
        
def clearCanvas():
    global maxX,maxY,minX,minY
    maxX = -1
    maxY = -1
    minX = 10000
    minY = 10000
    wn.delete('paint')
    
def saveLetter():
    global maxX,maxY,minX,minY
    global letter
    # if minY < 120 or maxY > 480:
    #     print(maxY," ",minY)
    #     print("try again")
    #     clearCanvas()
    #     return
    stroke = []
    stroke.append(letter)
    curStroke = []
    curStroke.append(letter[0])
    for i in range(1,len(letter),1):
        if abs(letter[i][0] - letter[i-1][0])+abs(letter[i][1] - letter[i-1][1]) > threshold:
            print("stroke found at ",i)
            stroke.append(curStroke)
            curStroke = []
        curStroke.append(letter[i])
    stroke.append(curStroke)
    
    
    tmpletter = []
    tmpstroke = []
    # tmpstroke.append(tmpletter)
    
    for letters in stroke:
        tmpletter = []
        preX = 1000
        preY = 1000
        for i in range(0,len(letters),1):
            x = letters[i][0]
            y = letters[i][1]
            if math.sqrt((preX-x)*(preX-x)+(preY-y)*(preY-y)) < 25:
                continue
            preX = x
            preY = y
            # d = 1.6
            tmpletter.append(letters[i])
            # wn.create_oval((x-10)/d, (y-10)/d, (x+10)/d, b(y+10)/d, fill='yellow', outline='yellow',tags='debug')
        tmpstroke.append(tmpletter)
        
    stroke = tmpstroke
    
    # add = 0
    # for letters in stroke:
    #     d = 1
    #     add = add+150
    #     for point in letters:
    #         x = point[0]
    #         y = point[1]
    #         wn.create_oval(add+(x-10)/d, (y-10)/d, add+(x+10)/d, (y+10)/d, fill='yellow', outline='yellow',tags='debug')

    
    lines = set()
    with open(r'letters/letterlist.txt') as f:
        for row in f:
            lines.add(row.rstrip('\n'))
    lines.add(making)
    f = open('letters/letterlist.txt', "w")
    for row in lines:
        f.write(row + "\n")
    f.close()    
    
    filename = "letters/"+making+".txt"
    print("_________________________"+filename)
    f = open(filename, "w")
    for letters in stroke:
        # print(letters)
        for point in letters:
            f.write(str(point[0])+" "+str(point[1])+" ")
        f.write("\n")
    f.close()
    print("file saved")

    #wn.delete('guideline')

    x, y = background.winfo_rootx()+122, background.winfo_rooty()
    w, h = background.winfo_width()-122, background.winfo_height()
    pyautogui.screenshot('images/'+making+'.jpg', region=(x, y, w, h))
    image = Image.open('images/'+making+'.jpg')
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('images/'+making+'.jpg')

    ex_crop_img = crop_contour( cv2.imread('images/'+making+'.jpg'), True)
    cv2.imwrite('images/'+making+'.jpg',ex_crop_img)

    # for j in range(150,550,100):
    #     gcl = 'blue'
    #     if j == 150 or j==450:
    #         gcl = 'red'
    #     for i in range(850):
    #         wn.create_oval(i, j, i+1, j+1, fill=gcl, outline=gcl,tags='guideline')

    clearCanvas()

def paint(event):
    global maxX,maxY,minX,minY
    global letter
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    # print(x1,y1)
    color = "black"
    maxX = max(maxX,max(x1,x2))
    minX = min(minX,min(x1,x2))
    maxY = max(maxY,max(y1,y2))
    minY = min(minY,min(y1,y2))
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')
    letter.append([x1,y1])
    

    
background=Canvas(root, width=970, height=700, bg='#eeeeee')
wn=Canvas(root, width=850, height=700, bg='white')
# for j in range(150,550,100):
#     gcl = 'blue'
#     if j == 150 or j==450:
#         gcl = 'red'
#     for i in range(850):
#         wn.create_oval(i, j, i+1, j+1, fill=gcl, outline=gcl,tags='guideline')

background.create_window(60,  50,window=Button(root,text='Practice', command=gotoPractice, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 190,window=Button(root,text='Test', command=gotoTest, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 120,window=Button(root,text='Create', command=gotoCreate, bg='#eeeeee', fg='black', font=('helvetica', 15, 'bold')))
background.create_window(60, 260,window=Button(root,text='clear', command=clearCanvas, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
label = Label(background, text = "Select the Letter :", font = ("Times New Roman", 9))
label.configure(background='#eeeeee')
label.place(x=5,y=290)
txtfld=Entry(background,width = 10)
txtfld.place(x=5, y=320)
button  = ttk.Button(root,text='ok', command=getLetter, padding=0, width=2,)

background.create_window(85, 332,window=button)

background.create_window(60, 380,window=Button(root,text='Save', command=saveLetter, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
wn.bind('<B1-Motion>', paint)
wn.place(x=120, y=0)
background.pack()


root.mainloop()

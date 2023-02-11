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


previous_tick = datetime.now()
previous_ticks = timer()
should_speak = True
current_language = "bangla"
mp3_file = ""
voice_thesh = 2

def justify_voice():
    global previous_tick,should_speak,voice_thesh
    current_tick = datetime.now()
    elapsed = current_tick-previous_tick
    elapsed_s = elapsed.total_seconds()
    # print(elapsed.total_seconds())
    if float(elapsed_s) >= voice_thesh:
        should_speak = True
    else:
        should_speak = False

    if should_speak:
        previous_tick = current_tick


def current_time():
    justify_voice()
    return str(datetime.now().time())[:-3]

currentTime = current_time()
timeTracker = {currentTime:False}


# model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2" # Link to model weights 
# IMAGE_SHAPE = (224, 224)

# layer = hub.KerasLayer(model_url, input_shape=IMAGE_SHAPE+(3,)) # Loading model weights
# model = tf.keras.Sequential([layer]) # Building model object using loaded weights


recognize=sr.Recognizer()




def crop_contour(image, plot=False):
    
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

# main.py 
currentLetter='A'
dotSize = 5
threshold = 15
index = 0
strokeIndex = 1
strokePointIndex = 0
root = Tk()
rootDestroyed = False
root.title("Home")
fontsizeC = ttk.Combobox()
letterChoice = ttk.Combobox()
select_letter = Label()
select_fontsize = Label()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
geometry = str(screen_width)
geometry = geometry+"x"
geometry = geometry+str(screen_height)
root.geometry(geometry)
session = 'Home'



def init():
    # return 
    global create_txtfld
    global fontsizeC
    global letterChoice
    global select_letter
    global select_fontsize,test_prev_x,test_prev_y
    test_prev_x = -1
    test_prev_y = -1
    create_txtfld.destroy()
    fontsizeC.destroy()
    letterChoice.destroy()
    select_letter.destroy()
    select_fontsize.destroy()
    background.delete('create_ok')
    background.delete('create_save')
    background.delete('test_button_evaluate')

    create_btn['background'] = 'brown'
    create_btn['foreground'] = 'white'
    home_btn['background'] = 'brown'
    home_btn['foreground'] = 'white'
    practice_btn['background'] = 'brown'
    practice_btn['foreground'] = 'white'
    test_btn['background'] = 'brown'
    test_btn['foreground'] = 'white'
    clear_btn['background'] = 'brown'
    clear_btn['foreground'] = 'white'

    



# practice.py

practice_currentLetter='capA'
practice_dotSize = 5
practice_threshold = 13
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
practice_letterChoiceV='capA'
practice_fontsizeV = 1


def practice_makeDataset():

    global wn
    global practice_dataset
    global practice_letter
    global practice_currentLetter
    global practice_CurrentSizeD
    global practice_index,practice_strokePointIndex,practice_strokeIndex
    global practice_started
    global practice_threshold
    global practice_dotSize,command,say,mp3_file
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
    global currentTime
    x1,y1,ovalStartx1,ovalStarty1 = x,y,tx,ty
    ovalEndx1,ovalEndy1 = tx,ty
    x2=x1+practice_threshold+5
    y2=y1+practice_threshold+5
    # print(" x1 = ",x1," y1 = ",y1," x2 = ",x2," y2 = ",y2," tx = ",tx," ty = ",ty)
    text = ""
    global command,mp3_file
    global say
    if x2 < tx and y2 < ty:
        text = "Go Down Right"
        mp3_file = "voice_folder/go_down_right.mp3"
    elif x1 > tx and y2 < ty:
        text = "Go Down Left"
        mp3_file = "voice_folder/go_down_left.mp3"
    elif x2 < tx and y1 > ty:
        text = "Go Up Right"
        mp3_file = "voice_folder/go_up_right.mp3"
    elif x1 > tx and y1 > ty:
        text = "Go Up Left"
        mp3_file = "voice_folder/go_up_left.mp3"
    elif y2 < ty and x1<=tx and x2>tx:
        text = "Go Down"
        mp3_file = "voice_folder/go_down.mp3"
    elif y1 > ty and x1<=tx and x2>tx:
        text = "Go Up"
        mp3_file = "voice_folder/go_up.mp3"
    elif x2 < tx and y1<=ty and y2>ty:
        text = "Go Right"
        mp3_file = "voice_folder/go_right.mp3"
    elif x1 > tx and y1<=ty and y2>ty:
        text = "Go Left"
        mp3_file = "voice_folder/go_left.mp3"
    
    if len(text) == 0:
        text = "keep on going"
        mp3_file = "voice_folder/keep_going.mp3"
        
    # new_time = current_time()
    # if should_speak:
    #     command=text
    #     say=True
    #     currentTime = new_time
    return text



def practice_responseNext(x,y,tx,ty):
    global currentTime,mp3_file
    x1,y1,ovalStartx1,ovalStarty1 = x,y,tx,ty
    ovalEndx1,ovalEndy1 = tx,ty
    x2=x1+practice_threshold  # practice_threshold has been determined after experimentation
    y2=y1+practice_threshold
    # print(" x1 = ",x1," y1 = ",y1," x2 = ",x2," y2 = ",y2," tx = ",tx," ty = ",ty)
    text = ""
    global command
    global say
    if x2 < tx and y2 < ty:
        text = "Go Down Right"
        mp3_file = "voice_folder/go_down_right.mp3"
    elif x1 > tx and y2 < ty:
        text = "Go Down Left"
        mp3_file = "voice_folder/go_down_left.mp3"
    elif x2 < tx and y1 > ty:
        text = "Go Up Right"
        mp3_file = "voice_folder/go_up_right.mp3"
    elif x1 > tx and y1 > ty:
        text = "Go Up Left"
        mp3_file = "voice_folder/go_up_left.mp3"
    elif y2 < ty and x1<=tx and x2>tx:
        text = "Go Down"
        mp3_file = "voice_folder/go_down.mp3"
    elif y1 > ty and x1<=tx and x2>tx:
        text = "Go Up"
        mp3_file = "voice_folder/go_up.mp3"
    elif x2 < tx and y1<=ty and y2>ty:
        text = "Go Right"
        mp3_file = "voice_folder/go_right.mp3"
    elif x1 > tx and y1<=ty and y2>ty:
        text = "Go Left"
        mp3_file = "voice_folder/go_left.mp3"
    
    if len(text) == 0:
        text = "keep on going"
        mp3_file = "voice_folder/keep_going.mp3"
    
    # new_time = current_time()
    # if should_speak:
    #     command=text
    #     currentTime = new_time
    #     say = True
    return text



def practice_getData(event):
    global letterChoice
    global practice_currentLetter
    global command,say,currentTime,mp3_file
    
    
    newLetter = letterChoice.get()
    if newLetter != practice_currentLetter:
        practice_currentLetter = newLetter
        practice_makeDataset()
        command = "letter chosen "+str(practice_currentLetter)+" font size selected "+str(practice_CurrentSizeD)
        mp3_file = "voice_folder/selection.mp3"
        
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()
        
        
        
def practice_getSize(event):
    global currentTime
    global practice_dataset
    global fontsizeC
    global practice_CurrentSizeD,command,say,mp3_file
    data = fontsizeC.current()
    divisors = [4,2,(4/3),1,0.8,(2/3),(4/7)]
    newSizeD = divisors[data]
        
    if newSizeD != practice_CurrentSizeD:
        practice_CurrentSizeD = newSizeD
        practice_makeDataset()
        command = "letter chosen "+str(practice_currentLetter)+" font size selected "+str(practice_CurrentSizeD)
        mp3_file = "voice_folder/selection.mp3"
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()

def practice_getDataV():
    global practice_letterChoiceV
    global practice_currentLetter,command,say,currentTime,mp3_file
    
    
    newLetter = practice_letterChoiceV
    if newLetter != practice_currentLetter:
        practice_currentLetter = newLetter
        practice_makeDataset()
        command = "letter chosen "+str(practice_currentLetter)+" font size selected "+str(practice_CurrentSizeD)
        mp3_file = "voice_folder/selection.mp3"
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()
        
def practice_getSizeV():
    global practice_dataset
    global practice_fontsizeV,currentTime
    global practice_CurrentSizeD,command,say,mp3_file
    data = practice_fontsizeV
    divisors = [4,2,(4/3),1,0.8,(2/3),(4/7)]
    newSizeD = divisors[data]
        
    if newSizeD != practice_CurrentSizeD:
        practice_CurrentSizeD = newSizeD
        practice_makeDataset()
        command = "letter chosen "+str(practice_currentLetter)+" font size selected "+str(practice_CurrentSizeD)
        mp3_file = "voice_folder/selection.mp3"
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()




# create

create_txtfld = Entry()
create_stroke_fld = Entry()

create_currentLetter='capA'
create_dotSize = 2
create_threshold = 20 
create_letterri = []
create_index = 0
create_letter = []
create_making = "capA"
create_maxX = -1
create_maxY = -1
create_minX = 10000
create_minY = 10000





def create_clearCanvas():
    global create_maxX,create_maxY,create_minX,create_minY,wn,create_letter
    create_maxX = -1
    create_maxY = -1
    create_minX = 10000
    create_minY = 10000
    wn.delete('paint')
    wn.delete('follow')
    create_letter.clear()

def create_getLetter():
    global create_txtfld
    print(type(create_txtfld))
    global create_making
    newLetter = create_txtfld.get()
    if len(newLetter) == 0:
        return
    print("here "+newLetter)
    create_making = newLetter.strip()
    if len(create_making)==1:
        if create_making[0] <= 'Z' and create_making[0] >= 'A':
            create_making = "cap"+create_making
        else:
            create_making = "SML"+create_making

    print(create_making)


def create_erosion_image(image_file,shift):
    image = cv2.imread(image_file)
    kernel = np.ones((shift,shift),np.uint8)
    image2 = cv2.erode(image,kernel,iterations = 1)
    cv2.imwrite("augmented.jpg", image2)
    # cv2.imwrite("augmented_image_part4" + "/C_erote-" + str(shift) + ".jpg", image2)
    # return test_evaluate_for_one()
    return image2

def create_dilation_image(image_file,shift):
    image = cv2.imread(image_file)
    kernel = np.ones((shift, shift), np.uint8)
    image2 = cv2.dilate(image,kernel,iterations = 1)
    cv2.imwrite("augmented.jpg", image2)
    # # cv2.imwrite("augmented_image_part4" + "/C_dialate-" + str(shift) + ".jpg", image2)
    # return test_evaluate_for_one()
    return image2

def create_rotate_image(image_file,deg):
    image = cv2.imread(image_file)
    rows, cols,c = image.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
    image2 = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite("augmented.jpg", image2)
    # # cv2.imwrite("augmented_image_part4" + "/C_rotate-" + str(deg) + ".jpg", image2)
    # return test_evaluate_for_one()
    return image2

def create_scale_image(image_file,fx,fy):
    image = cv2.imread(image_file)
    image2 = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("augmented.jpg", image2)
    # # cv2.imwrite("augmented_image_part4" + "/C_scale-" + str(fx)+ "_" +str(fy) + ".jpg", image2)
    # return test_evaluate_for_one()
    return image2


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

    pick = open('svm.sav','wb')
    pickle.dump(model,pick)
    pick.close()




def create_saveLetter():
    global create_maxX,create_maxY,create_minX,create_minY
    global create_letter,stc,wn
    global background
    if len(create_letter) == 0:
        return 
    # if create_minY < 120 or create_maxY > 480:
    #     print(create_maxY," ",create_minY)
    #     print("try again")
    #     create_clearCanvas()
    #     return
    stroke = []
    stroke.append(create_letter)
    curStroke = []
    curStroke.append(create_letter[0])
    for i in range(1,len(create_letter),1):
        if abs(create_letter[i][0] - create_letter[i-1][0])+abs(create_letter[i][1] - create_letter[i-1][1]) > create_threshold:
            print("stroke found at ",i)
            stroke.append(curStroke)
            curStroke = []
        curStroke.append(create_letter[i])
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
            if math.sqrt((preX-x)*(preX-x)+(preY-y)*(preY-y)) < 25 and stc:
                continue
            preX = x
            preY = y
            # d = 1.6
            tmpletter.append(letters[i])
            # wn.create_oval((x-10)/d, (y-10)/d, (x+10)/d, b(y+10)/d, fill='yellow', outline='yellow',tags='debug')
        tmpstroke.append(tmpletter)
        
    stroke = tmpstroke


    base_dir = 'image_dataset/'
    train_dir = os.path.join(base_dir, create_making)
    os.listdir(base_dir)
    list = os.listdir(base_dir)
    if create_making in list: 
        #print(list)
        pass
    else: 
        os.mkdir(train_dir)

    x, y = background.winfo_rootx()+122, background.winfo_rooty()
    w, h = background.winfo_width()-130, background.winfo_height()-77
    pyautogui.screenshot('images/'+create_making+'.jpg', region=(x, y, w, h))
    image = Image.open('images/'+create_making+'.jpg')
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('images/'+create_making+'.jpg')

    ex_crop_img = crop_contour( cv2.imread('images/'+create_making+'.jpg'), True)
    cv2.imwrite('images/'+create_making+'.jpg',ex_crop_img)
    cv2.imwrite('image_dataset/'+create_making+'/'+create_making+'_Standard.jpg',ex_crop_img)
    print('Hello at create evaluate')

    image ='image_dataset/'+create_making+'/'+create_making+'_Standard.jpg'

    cnt = 1
    for i in np.arange(-5, 5, 0.5):
        new_img=create_rotate_image(image,i)
        cv2.imwrite('image_dataset/'+create_making+'/'+create_making+'_Rotate'+str(cnt)+'.jpg',new_img)
        cnt = cnt+1

    cnt = 1
    for i in np.arange(0.1, 2, 0.5):
        for j in np.arange(0.1, 2, 0.5):
            new_img=create_scale_image(image,i,j)
            cv2.imwrite('image_dataset/'+create_making+'/'+create_making+'_Scale'+str(cnt)+'.jpg',new_img)
            cnt=cnt+1

    for i in range(0, 10, 1):
        new_img=create_erosion_image(image,i)
        cv2.imwrite('image_dataset/'+create_making+'/'+create_making+'_Erode'+str(cnt)+'.jpg',new_img)
        cnt = cnt+1

    for i in range(0, 10, 1):
        new_img=create_dilation_image(image,i)
        cv2.imwrite('image_dataset/'+create_making+'/'+create_making+'_Dilate'+str(cnt)+'.jpg',new_img)
        cnt = cnt+1

    
    # Machine Learning --------------------
    create_ML_Model()
    
    
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
    lines.add(create_making)
    f = open('letters/letterlist.txt', "w")
    for row in lines:
        f.write(row + "\n")
    f.close()    

    stc = False
    
    wn.delete('paint')
    filename = "letters/"+create_making+".txt"
    print("_________________________"+filename)
    f = open(filename, "w")

    for letters in stroke:
        # print(letters)
            
        for point in letters:
            f.write(str(point[0])+" "+str(point[1])+" ")
            if stc and False:
                color="blue"
                x1, y1 = (point[0]-5), (point[1]-5)
                x2, y2 = (point[0]+5), (point[1]+5)
                wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='strokes')
        print("found stroke")
        f.write("\n")
        # input("Press Enter to continue...")

        
        # wn.delete('strokes')
        # stc = True

    f.close()
    print("file saved")

    #wn.delete('guideline')

    
    # for j in range(150,550,100):
    #     gcl = 'blue'
    #     if j == 150 or j==450:
    #         gcl = 'red'
    #     for i in range(850):
    #         wn.create_oval(i, j, i+1, j+1, fill=gcl, outline=gcl,tags='guideline')

    create_clearCanvas()





# test.py

test_currentLetter='capA'
test_dotSize = 2
test_threshold = 20 
test_letterri = []
test_index = 0
test_prev_x = -1
test_prev_y = -1

def test_predict_with_model():
    global currentTime,say,command,text
    catagories = os.listdir('image_dataset/')
    path, dirs, files = next(os.walk("image_dataset/"))

    pick = open('svm.sav','rb')
    model = pickle.load(pick)
    pick.close()

    imgpath = "screenshot.jpg"
    img = cv2.imread(imgpath,0)

    img = cv2.resize(img,(50,50))
    #plt.imshow(img,cmap='gray')
    image = np.array(img).flatten()
    confidence = model.predict_proba([image])
    print(confidence)
    prediction = model.predict([image])
    print("Prediction:",prediction)
    print("prediction :",catagories[prediction[0]])

    text = 'The letter is like '+str(catagories[prediction[0]])
    mp3_file = "voice_folder/coming_soon.mp3"
    command=text
    print(command)
    new_time = current_time()
    if should_speak:
        currentTime = new_time
        say = True
        t1= threading.Thread(target=voiceGuide, name='t1')
        t1.start()



def test_getData(event):
    global test_currentLetter,test_prev_x
    global letterChoice,command,say,test_prev_y,currentTime,mp3_file
    test_newLetter = letterChoice.get()
    
    if test_newLetter != test_currentLetter:
        test_currentLetter = test_newLetter
        wn.delete('all')
        print(test_currentLetter)
        command = "letter chosen "+str(test_currentLetter)
        mp3_file = "voice_folder/selection.mp3"
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()
        test_prev_x = -1
        test_prev_y = -1

def test_getDataV(test_newLetter):
    global test_currentLetter,command,say,currentTime,mp3_file
    global test_prev_x 
    global test_prev_y 
    if test_newLetter != test_currentLetter:
        test_currentLetter = test_newLetter
        wn.delete('all')
        print(test_currentLetter)
        command = "letter chosen "+str(test_currentLetter)
        mp3_file = "voice_folder/selection.mp3"
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()
        test_prev_x = -1
        test_prev_y = -1



# def test_evaluate_for_one():
#     global command,say,test_currentLetter,currentTime

#     print("coming soon!!!")
#     x, y = background.winfo_rootx()+122, background.winfo_rooty()
#     w, h = background.winfo_width()-130, background.winfo_height()-77
#     pyautogui.screenshot('screenshot.jpg', region=(x, y, w, h))
#     image = Image.open('screenshot.jpg')
#     inverted_image = PIL.ImageOps.invert(image)
#     inverted_image.save('screenshot.jpg')


#     ex_crop_img = crop_contour( cv2.imread('screenshot.jpg'), True)
#     cv2.imwrite('screenshot.jpg',ex_crop_img)

#     test_predict_with_model()
    
#     img_1='screenshot.jpg'
#     img_2='augmented.jpg'
#     #img_2='images/'+test_currentLetter+'.jpg'

#     # Calculations for image 1
#     img_1 = Image.open(img_1).convert('L').resize(IMAGE_SHAPE) # Resizing the image to required size
#     # img_1.save('img_1.jpg')
#     img_1 = np.stack((img_1,)*3, axis=-1) # Converting the image into a color representation for each pixel
#     img_1 = np.array(img_1)/255.0 # Normalizing the values between 0 and 1

#     # Average Blur
#     img_1 = cv2.blur(img_1,(3,3))

#     embedding_img1 = model.predict(img_1[np.newaxis, ...]) # Extracting the features
#     embedding_img1_np = np.array(embedding_img1) # Converting to numpy array
#     flattened_feature_img1 = embedding_img1_np.flatten() # Converting matrix to a vector

#     # Calculations for image 2
#     img_2 = Image.open(img_2).convert('L').resize(IMAGE_SHAPE) # Resizing the image to required size
#     # img_2.save('img_2.jpg')
#     img_2 = np.stack((img_2,)*3, axis=-1) # Converting the image into a color representation for each pixel
#     img_2 = np.array(img_2)/255.0 # Normalizing the values between 0 and 1

#     # Average Blur
#     img_2 = cv2.blur(img_2,(3,3))

#     embedding_img2 = model.predict(img_2[np.newaxis, ...]) # Extracting the features
#     embedding_img2_np = np.array(embedding_img2) # Converting to numpy array
#     flattened_feature_img2 = embedding_img2_np.flatten() # Converting matrix to a vector

#     methods = ['sqeuclidean', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
#         'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
#         'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
#         'sokalmichener', 'sokalsneath', 'braycurtis', 'yule']

#     mins = [0,0,0,0,0,0,0,0,0,0,0,0.475,0,0,0,0.475,0,0,0,0]
#     maxes = [150,450, 2.5, 200, .5, .4, .4, 13, .6, 1, 0.5, 1, .6, 13, .4, .8, .5, .7, .5, .3]
#     scores = []

#     for i, m in enumerate(methods):

#         metric = m # Try using any one of the methods listed above if needed
#         dist_boyboy = distance.cdist([flattened_feature_img1], [flattened_feature_img2],metric)[0]      # Finding similarity 

#         score = max(0, maxes[i] - dist_boyboy[0])
#         #print(score)
#         interval = maxes[i] - mins[i]
#         score = score / interval * 100
#         scores.append(score)

#     return max(scores)

# def erosion_image(image_file,shift):
#     image = cv2.imread(image_file)
#     kernel = np.ones((shift,shift),np.uint8)
#     image2 = cv2.erode(image,kernel,iterations = 1)
#     cv2.imwrite("augmented.jpg", image2)
#     # cv2.imwrite("augmented_image_part4" + "/C_erote-" + str(shift) + ".jpg", image2)
#     return test_evaluate_for_one()

# def dilation_image(image_file,shift):
#     image = cv2.imread(image_file)
#     kernel = np.ones((shift, shift), np.uint8)
#     image2 = cv2.dilate(image,kernel,iterations = 1)
#     cv2.imwrite("augmented.jpg", image2)
#     # cv2.imwrite("augmented_image_part4" + "/C_dialate-" + str(shift) + ".jpg", image2)
#     return test_evaluate_for_one()

# def rotate_image(image_file,deg):
#     image = cv2.imread(image_file)
#     rows, cols,c = image.shape
#     M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
#     image2 = cv2.warpAffine(image, M, (cols, rows))
#     cv2.imwrite("augmented.jpg", image2)
#     # cv2.imwrite("augmented_image_part4" + "/C_rotate-" + str(deg) + ".jpg", image2)
#     return test_evaluate_for_one()

# def scale_image(image_file,fx,fy):
#     image = cv2.imread(image_file)
#     image2 = cv2.resize(image,None,fx=fx, fy=fy, interpolation = cv2.INTER_CUBIC)
#     cv2.imwrite("augmented.jpg", image2)
#     # cv2.imwrite("augmented_image_part4" + "/C_scale-" + str(fx)+ "_" +str(fy) + ".jpg", image2)
#     return test_evaluate_for_one()


def test_evaluate():

    global currentTime,say,command,text,mp3_file

    x, y = background.winfo_rootx()+122, background.winfo_rooty()
    w, h = background.winfo_width()-130, background.winfo_height()-77
    pyautogui.screenshot('screenshot.jpg', region=(x, y, w, h))
    image = Image.open('screenshot.jpg')
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('screenshot.jpg')


    ex_crop_img = crop_contour( cv2.imread('screenshot.jpg'), True)
    cv2.imwrite('screenshot.jpg',ex_crop_img)

    test_predict_with_model()

    # src='images/'+test_currentLetter+'.jpg'
    # standard_img=cv2.imread('images/'+test_currentLetter+'.jpg')
    # cv2.imwrite('augmented.jpg',standard_img)

    # scores = []
    # scores.append(test_evaluate_for_one())
    # print("standard",str(scores[len(scores) - 1]))
    # cnt = 0
    # for i in np.arange(-5, 5, 0.5):
    #     scores.append(rotate_image(src,i))
    #     # print(str(scores[len(scores) - 1]))
    #     cnt = cnt+1
    #     print("rotation",i,cnt,str(scores[len(scores) - 1]))

    # for i in np.arange(0.1, 2, 0.5):
    #     for j in np.arange(0.1, 2, 0.5):
    #         scores.append(scale_image(src,i,j))
    #         # print(str(scores[len(scores) - 1]))
    #         cnt = cnt+1
    #         print("scale",i,j,cnt,str(scores[len(scores) - 1]))

    # for i in range(0, 10, 1):
    #     scores.append(erosion_image(src,i))
    #     # print(str(scores[len(scores) - 1]))
    #     cnt = cnt+1
    #     print("erote",i,cnt,str(scores[len(scores) - 1]))

    # for i in range(0, 10, 1):
    #     scores.append(dilation_image(src,i))
    #     # print(str(scores[len(scores) - 1]))
    #     cnt = cnt+1
    #     print("dialate",i,cnt,str(scores[len(scores) - 1]))

    # text = 'Similarity is '+str(round(max(scores),2))+' %'
    # mp3_file = "voice_folder/coming_soon.mp3"
    # command=text
    # print(command)
    # new_time = current_time()
    # if should_speak:
    #     currentTime = new_time
    #     say = True
    #     t1= threading.Thread(target=voiceGuide, name='t1')
    #     t1.start()

def test_slope(x,px,y,py):
    return (y-py)/(x-px)

def test_getY(x,px,y,py,x2):
    m = test_slope(x,px,y,py)
    y = py+m*(x2-px)
    return y





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
    global command,mp3_file
    global say,currentTime
    if practice_index == len(practice_letter):
        command='Done'
        mp3_file = "voice_folder/done.mp3"
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()
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
            if xc>screen_width-130:
                print("too much on right side, go left")
                command = "too much on right side, go left"
                mp3_file = "voice_folder/init_left.mp3"
                new_time = current_time()
                if should_speak:
                    currentTime = new_time
                    say = True
                    t1= threading.Thread(target=voiceGuide, name='t1')
                    t1.start()
                practice_makeDataset()
                break
            if xc<10 :
                print("too much on left side, go right")
                command = "too much on left side, go right"
                mp3_file = "voice_folder/init_right.mp3"
                new_time = current_time()
                if should_speak:
                    currentTime = new_time
                    say = True
                    t1= threading.Thread(target=voiceGuide, name='t1')
                    t1.start()
                practice_makeDataset()
                break
            if yc<10 :
                print("too much on upperside, go down")
                command = "too much on upperside, go down"
                mp3_file = "voice_folder/init_down.mp3"
                new_time = current_time()
                if should_speak:
                    currentTime = new_time
                    say = True
                    t1= threading.Thread(target=voiceGuide, name='t1')
                    t1.start()
                practice_makeDataset()
                break
            if yc>screen_height-10 :
                print("too much on bottom side, go up")
                command = "too much on bottom side, go up"
                mp3_file = "voice_folder/init_up.mp3"
                new_time = current_time()
                if should_speak:
                    currentTime = new_time
                    say = True
                    t1= threading.Thread(target=voiceGuide, name='t1')
                    t1.start()
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
            text = practice_correctionMSGpoint(x,y,pretX,pretY)
            print(text)
            command=text
            new_time = current_time()
            if should_speak:
                currentTime = new_time
                say = True
                t1= threading.Thread(target=voiceGuide, name='t1')
                t1.start()
            draw_point = False
    
    
    if draw_point:
        wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')
        text = practice_responseNext(x,y,targetX,targetY)
        print(text)
        command=text
        new_time = current_time()
        if should_speak:
            currentTime = new_time
            say = True
            t1= threading.Thread(target=voiceGuide, name='t1')
            t1.start()
        

        


def paintC(event):
    
    global create_maxX,create_maxY,create_minX,create_minY
    global create_letter,currentTime
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    # print(x1,y1)
    color = "black"
    create_maxX = max(create_maxX,max(x1,x2))
    create_minX = min(create_minX,min(x1,x2))
    create_maxY = max(create_maxY,max(y1,y2))
    create_minY = min(create_minY,min(y1,y2))
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')
    create_letter.append([x1,y1])

def paintT(event):
    # get x1, y1, x2, y2 co-ordinates
    x1, y1 = (event.x-5), (event.y-5)
    x2, y2 = (event.x+5), (event.y+5)
    color = "black"
    # display the mouse movement inside canvas
    
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')        
    
    









background=Canvas(root, width=screen_width, height=screen_height, bg='#eeeeee')


def selectLanguageBangla():
    global current_language,banglaSelect,englishSelect
    current_language = "bangla"
    banglaSelect['background'] = 'white'
    banglaSelect['foreground'] = 'black'
    englishSelect['background'] = 'brown'
    englishSelect['foreground'] = 'white'

def selectLanguageEnglish():
    global current_language,englishSelect,banglaSelect
    current_language = "english"
    englishSelect['background'] = 'white'
    englishSelect['foreground'] = 'black'
    banglaSelect['background'] = 'brown'
    banglaSelect['foreground'] = 'white'


def gotoHome():
    global session
    global background,currentTime
    global create_txtfld,command,say,mp3_file
    session = 'Home'
    init()
    command = 'Home ready'
    mp3_file = "voice_folder/home_ready.mp3"
    new_time = current_time()
    if should_speak:
        currentTime = new_time
        say = True
        t1= threading.Thread(target=voiceGuide, name='t1')
        t1.start()
    wn.delete('paint')
    wn.delete('follow')
    home_btn['background'] = 'white'
    home_btn['foreground'] = 'black'
    

def gotoPractice():
    global session,command,say,mp3_file
    global letterChoice
    global fontsizeC
    global select_letter
    global select_fontsize,currentTime
    strrng = StringVar() 
    init()
    practice_makeDataset()
    command = 'Practice ready'
    mp3_file = "voice_folder/practice_ready.mp3"
    # command=কবো নাকো বদ্ধ ঘরে"

    
    new_time = current_time()
    if should_speak:
        currentTime = new_time
        say = True
        t1= threading.Thread(target=voiceGuide, name='t1')
        t1.start()
    practice_btn['background'] = 'white'
    practice_btn['foreground'] = 'black'
    select_letter = Label(background, text = "Select Letter :", font = ("Times New Roman", 12))
    select_letter.configure(background='#eeeeee')
    select_letter.place(x=5,y=360)
    letterChoice = ttk.Combobox(background, width = 10, textvariable = strrng) 

    lines = []
    with open(r'letters/letterlist.txt') as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    letterChoice['values'] = lines
        
    letterChoice.bind("<<ComboboxSelected>>",practice_getData)
    letterChoice.current(0)
    letterChoice.place(x=5, y=385)

    strrng = StringVar() 
    select_fontsize = Label(background, text = "Select size :", font = ("Times New Roman", 12))
    select_fontsize.configure(background='#eeeeee')
    select_fontsize.place(x=5,y=410)
    fontsizeC = ttk.Combobox(background, width = 10, textvariable = strrng) 

    lines = ['0(25%)','1(50%)','2(75%)','3(100%)','4(125%)','5(150%)','6(175%)']
    fontsizeC['values'] = lines
        
    fontsizeC.bind("<<ComboboxSelected>>",practice_getSize)
    fontsizeC.current(3)
    fontsizeC.place(x=5, y=435)

    practice_makeDataset()
    session = 'Practice'


def gotoTest():
    global session,command,say,currentTime,mp3_file
    global letterChoice
    global select_letter,test_prev_x,test_prev_y
    session = 'Test'
    
    init()
    wn.delete('paint')
    wn.delete('follow')
    command = 'Test ready'
    mp3_file = "voice_folder/test_ready.mp3"
    new_time = current_time()
    if should_speak:
        currentTime = new_time
        say = True
        t1= threading.Thread(target=voiceGuide, name='t1')
        t1.start()
    test_btn['background'] = 'white'
    test_btn['foreground'] = 'black'
    select_letter = Label(background, text = "Select Letter :", font = ("Times New Roman", 12))
    select_letter.configure(background='#eeeeee')
    select_letter.place(x=5,y=360)
    strrng = StringVar()
    letterChoice = ttk.Combobox(background, width = 10, textvariable = strrng) 
    lines = []
    with open(r'letters/letterlist.txt') as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    letterChoice['values'] = lines
        
    letterChoice.bind("<<ComboboxSelected>>",test_getData)
    letterChoice.current(0)
    letterChoice.place(x=5, y=385)
    
    background.create_window(60, 450,window=Button(root,text='Evaluate', command=test_evaluate, bg='brown', fg='white', font=('helvetica', 12, 'bold')),tags = 'test_button_evaluate')

def gotoCreate():
    global select_letter,command,say,mp3_file
    global create_txtfld
    global session
    global background,currentTime
    session = 'Create'

    init()
    create_clearCanvas()
    command = 'create ready'
    mp3_file = "voice_folder/create_ready.mp3"
    new_time = current_time()
    if should_speak:
        currentTime = new_time
        say = True
        t1= threading.Thread(target=voiceGuide, name='t1')
        t1.start()
    create_btn['background'] = 'white'
    create_btn['foreground'] = 'black'
    select_letter = Label(background, text = "Select Letter :", font = ("Times New Roman", 12))
    select_letter.configure(background='#eeeeee')
    select_letter.place(x=0,y=360)
    create_txtfld=Entry(background,width = 10)
    create_txtfld.place(x=5, y=390)
    background.create_window(105, 400,window=Button(root,text='ok', command=create_getLetter, bg='white', fg='black', font=('helvetica', 7, 'bold')),tags='create_ok')
    background.create_window(60, 460,window=Button(root,text='Save', command=create_saveLetter, bg='brown', fg='white', font=('helvetica', 12, 'bold')),tags ='create_save')


def clear():
    global wn,command,say,currentTime,mp3_file
    if session == 'Home':
        wn.delete('paint')
        wn.delete('follow')
    elif session == 'Practice':
        practice_makeDataset()
    elif session == 'Create':
        create_clearCanvas()
    elif session == 'Test':
        wn.delete('paint')
        wn.delete('follow')
    command='screen cleared'
    mp3_file = "voice_folder/cleared.mp3"
    new_time = current_time()
    if should_speak:
        currentTime = new_time
        say = True
        t1= threading.Thread(target=voiceGuide, name='t1')
        t1.start()
















# voiceGuide _____________________________________________

def voiceGuide():
    #return 
    global say
    global command,mp3_file
    global timeTracker
    global current_language,mp3_file,previous_ticks,voice_thesh
    lim = 1
    print("Wasif inside voiceGuide at ",currentTime)
    try:
        if(say==True):
            curTime = timer()
            print("cur=",curTime," prev=",previous_ticks," elapsed = ",curTime-previous_ticks)
            if(curTime-previous_ticks) < voice_thesh:
                print("passing ",curTime," ",previous_ticks)
                pass
            previous_ticks = curTime
            if current_language == "english" :
                print(command," ________entered here",currentTime)
                if currentTime in timeTracker.keys():
                    return 
                print('Saying....')
                # Initialize the engine
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                # voiceFemales = filter(lambda v: v.gender == 'VoiceGenderFemale', voices)
                engine.setProperty('voice', voices[0].id)
                engine.say(command)
                engine.runAndWait()
                
            else:
                # print("the mp3 file is")
                # print(mp3_file)
                # playsound(mp3_file)

                mixer.init()
                mixer.music.load(mp3_file)
                mixer.music.play()
                while mixer.music.get_busy():  # wait for music to finish playing
                    time.sleep(1)

            say=False
            timeTracker[currentTime] = True
            
    except:
        say=False
        return 

    print("voice end")   
    print('number of current threads is ', threading.active_count())







# Speech Detection __________________________________________________________

def getAudio():
    return 
    global practice_letterChoiceV
    global practice_fontsizeV
    while rootDestroyed == False:
        print("wasif inside audio")
        try:
            with sr.Microphone() as mic:
                print("entered!!")
                recognize.adjust_for_ambient_noise(mic,duration=0.1)
                audio=recognize.listen(mic)

                text=recognize.recognize_google(audio)
                text=text.lower()
                
                print(f"Recognized the Speech : {text}")

                if text == "go to home":
                    gotoHome()
                elif text == "go to create":
                    gotoCreate()
                elif text == "go to test":
                    gotoTest()
                elif text == "go to practice":
                    gotoPractice()
                elif text == "clear screen":
                    clear()
                elif session == 'Test' and text == "evaluate":
                    test_evaluate()
                else:
                    if session != 'Home' and session != 'Create' and (text.find('write') != -1 or text.find('letter') != -1 or text.find('capital') != -1 or text.find('small') != -1):
                        if text.find('letter') != -1:
                            if text.find('capital') != -1:
                                strv = text.split()
                                str = strv[-1]
                                str = "cap"+str.upper()
                                practice_letterChoiceV = str
                                test_newLetter = str
                                if session == 'Practice':
                                    practice_getDataV()
                                else:
                                    test_getDataV(test_newLetter)
                                
                            else:
                                strv = text.split()
                                str = strv[-1]
                                practice_letterChoiceV = str
                                test_newLetter = str
                                if session == 'Practice':
                                    practice_getDataV()
                                else:
                                    test_getDataV(test_newLetter)


                    elif session == 'Practice' and (text.find('font') != -1 or text.find('size') != -1 or text.find('set') != -1) :
                        strv = text.split()
                        str = strv[-1]
                        practice_fontsizeV = 0
                        if (str == "zero"):
                            practice_fontsizeV = 0
                        else:
                            practice_fontsizeV = int(str)
                        
                        practice_getSizeV()

        except :
            continue

    print("audio end")
    print('number of current threads is ', threading.active_count())





t2= threading.Thread(target=getAudio, name='t2')
t2.start()










stc = True
home_btn = Button(root,text='Home', command=gotoHome, bg='white', fg='black', font=('helvetica', 12, 'bold'))
practice_btn = Button(root,text='Practice', command=gotoPractice, bg='Brown', fg='white', font=('helvetica', 12, 'bold'))
test_btn = Button(root,text='Test', command=gotoTest, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
create_btn = Button(root,text='Create', command=gotoCreate, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
clear_btn = Button(root,text='clear', command=clear, bg='brown', fg='white', font=('helvetica', 12, 'bold'))
banglaSelect = Button(root,text='Bangla', command=selectLanguageBangla, bg='white', fg='black', font=('helvetica', 12, 'bold'))
englishSelect = Button(root,text='English', command=selectLanguageEnglish, bg='brown', fg='white', font=('helvetica', 12, 'bold'))


background.create_window(60,  50,height = 40,width = 100,window=home_btn)
background.create_window(60, 120,height = 40,width = 100,window=practice_btn)
background.create_window(60, 190,height = 40,width = 100,window=test_btn)
background.create_window(60, 260,height = 40,width = 100,window=create_btn)
background.create_window(60, 330,height = 40,width = 100,window=clear_btn)
background.create_window(60, 900,height = 40,width = 100,window=banglaSelect)
background.create_window(60, 950,height = 40,width = 100,window=englishSelect)



wn=Canvas(root, width=screen_width, height=screen_height, bg='white')
wn.bind('<B1-Motion>', paint)


wn.place(x=120, y=0)



background.pack()

root.mainloop()

rootDestroyed = True

print("wasif inside main")
print('number of current threads is ', threading.active_count())

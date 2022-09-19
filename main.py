from ctypes import alignment
import math
from mimetypes import init
from time import sleep
from tkinter import *
from tkinter import ttk 
import pyttsx3
import threading
from cgitb import text
from email.mime import audio
import speech_recognition as sr
import pyttsx3
import cv2
import imutils 
import pyautogui
from PIL import Image
import PIL.ImageOps  
  


def crop_contour(image, plot=False):
    
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray,5)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.create_threshold(gray, 46, 255, cv2.THRESH_BINARY)[1]
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



# create

create_currentLetter='A'
create_dotSize = 2
create_threshold = 20 
create_letterri = []
create_index = 0
create_letter = []
create_making = ""
create_maxX = -1
create_maxY = -1
create_minX = 10000
create_minY = 10000





def create_clearCanvas():
    global create_maxX,create_maxY,create_minX,create_minY
    create_maxX = -1
    create_maxY = -1
    create_minX = 10000
    create_minY = 10000
    wn.delete('paint')

def create_getLetter(txtfld):
    pass
    global create_making
    newLetter = txtfld.get()
    create_making = newLetter.strip()
    if create_making[0] <= 'Z' and create_making[0] >= 'A':
        create_making = "cap"+create_making
    else:
        create_making = "SML"+create_making

    print(create_making)

def create_saveLetter():
    global create_maxX,create_maxY,create_minX,create_minY
    global create_letter
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
    lines.add(create_making)
    f = open('letters/letterlist.txt', "w")
    for row in lines:
        f.write(row + "\n")
    f.close()    
    
    filename = "letters/"+create_making+".txt"
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
    pyautogui.screenshot('images/'+create_making+'.jpg', region=(x, y, w, h))
    image = Image.open('images/'+create_making+'.jpg')
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('images/'+create_making+'.jpg')

    ex_crop_img = crop_contour( cv2.imread('images/'+create_making+'.jpg'), True)
    cv2.imwrite('images/'+create_making+'.jpg',ex_crop_img)

    # for j in range(150,550,100):
    #     gcl = 'blue'
    #     if j == 150 or j==450:
    #         gcl = 'red'
    #     for i in range(850):
    #         wn.create_oval(i, j, i+1, j+1, fill=gcl, outline=gcl,tags='guideline')

    create_clearCanvas()

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
    
    global create_maxX,create_maxY,create_minX,create_minY
    global create_letter
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
    color = "green"
    # display the mouse movement inside canvas
    wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')












background=Canvas(root, width=screen_width, height=screen_height, bg='#eeeeee')

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
def gotoCreate():
    global session
    global background
    session = 'Create'
    label = Label(background, text = "Select the Letter :", font = ("Times New Roman", 12))
    label.configure(background='#eeeeee')
    label.place(x=5,y=400)
    txtfld=Entry(background,width = 10)
    txtfld.place(x=5, y=420)
    button  = ttk.Button(root,text='ok', command=create_getLetter(txtfld), padding=0, width=2,)

    # background.create_window(0, 440,window=button)

    # background.create_window(0, 470,window=Button(root,text='Save', command=create_saveLetter, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
    # background.create_window(60, 470,height = 40,width = 100,window=Button(root,text='clear', command=None, bg='brown', fg='white', font=('helvetica', 12, 'bold')))
    

def clear():
    pass


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




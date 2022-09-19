import math
from mimetypes import init
from tkinter import *
from tkinter import ttk 
from subprocess import Popen
from math import atan2, degrees, radians
import pyttsx3
import threading
from cgitb import text
from email.mime import audio
import speech_recognition as sr
import pyttsx3
import datetime

recognize=sr.Recognizer()

#testing git
#testing vcGIT

practice_currentLetter='A'
practice_dotSize = 5
practice_threshold = 15
practice_index = 0
practice_strokeIndex = 1
practice_strokePointIndex = 0
root = Tk()
root.title("Practice")

root.geometry("1000x1000")
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


def voiceGuide():
    global say
    global command
    while(1):
        # print("rrr")
        if(say==True):
            print('Saying....')
            # Initialize the engine
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            # voiceFemales = filter(lambda v: v.gender == 'VoiceGenderFemale', voices)
            engine.setProperty('voice', voices[0].id)
            engine.say(command)
            engine.runAndWait()
            say=False

# sr.UnknownValueError():

def getAudio():
    global practice_letterChoiceV
    global practice_fontsizeV
    while True:
        # print("wasif")
        try:
            with sr.Microphone() as mic:
                print("entered!!")
                recognize.adjust_for_ambient_noise(mic,duration=0.1)
                audio=recognize.listen(mic)

                text=recognize.recognize_google(audio)
                text=text.lower()
                
                print(f"Recognized the Speech : {text}")

                if text == "go to create":
                    gotoCreate()
                elif text == "go to test":
                    gotoTest()
                elif text == "go to practice":
                    gotoPractice()
                elif text == "clear screen":
                    clearCanvas()
                else:
                    if text.find('write') != -1 or text.find('practice_letter') != -1 or text.find('capital') != -1 or text.find('small') != -1:
                        if text.find('practice_letter') != -1:
                            if text.find('capital') != -1:
                                strv = text.split()
                                str = strv[-1]
                                str = "cap"+str.upper()
                                practice_letterChoiceV = str
                                practice_getDataV()
                            else:
                                strv = text.split()
                                str = strv[-1]
                                practice_letterChoiceV = str
                                practice_getDataV()

                    elif text.find('font') != -1 or text.find('size') != -1 or text.find('set') != -1 :
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

def gotoPractice():
    print("yo wassup")

def gotoTest():
    Popen(['python', 'test.py'])
    root.withdraw()
    
def gotoCreate():
    Popen(['python', 'create.py'])
    root.withdraw()
    
def getEuclid(x1,y1,x2,y2):
    return math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

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
       
    # add = 0
    # for letters in practice_dataset:
    #     add = add+150
    #     for point in letters:
    #         x = point[0]
    #         y = point[1]
    #         wn.create_oval(add+(x-10), (y-10), add+(x+10), (y+10), fill='yellow', outline='yellow',tags='debug')

    
def theta(x1,y1,x2,y2):
    angle = atan2(y2 - y1, x2 - x1)
    
    angle = degrees(angle)
    if angle < 0:
        angle = 360+angle
    return angle
    
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

def practice_getDataV():
    global practice_letterChoiceV
    global practice_currentLetter
    
    
    newLetter = practice_letterChoiceV
    if newLetter != practice_currentLetter:
        practice_currentLetter = newLetter
        practice_makeDataset()
        
def practice_getSizeV():
    global practice_dataset
    global practice_fontsizeV
    global practice_CurrentSizeD
    data = practice_fontsizeV
    divisors = [4,2,(4/3),1,0.8,(2/3),(4/7)]
    newSizeD = divisors[data]
        
    if newSizeD != practice_CurrentSizeD:
        practice_CurrentSizeD = newSizeD
        practice_makeDataset()

            

        
def clearCanvas():
    practice_makeDataset()


#####proper instructions

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




def paint(event):
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
        

     
    # addx = practice_initX
    # addy = practice_initY
    
    # for letters in practice_dataset:
    #     for point in letters:
    #         x = point[0]+addx
    #         y = point[1]+addy
    #         wn.create_oval((x-10), (y-10), (x+10), (y+10), fill='yellow', outline='yellow',tags='debug')
    #     break
    

t1= threading.Thread(target=voiceGuide, name='t1')
t1.start()

t2= threading.Thread(target=getAudio, name='t2')
t2.start()
    
background=Canvas(root, width=2000, height=2000, bg='#eeeeee')
wn=Canvas(root, width=500, height=2000, bg='white')


background.create_window(60,  50,window=Button(root,text='Practice', command=gotoPractice, bg='#eeeeee', fg='black', font=('helvetica', 15, 'bold')))
background.create_window(60, 190,window=Button(root,text='Test', command=gotoTest, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 120,window=Button(root,text='Create', command=gotoCreate, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 260,window=Button(root,text='clear', command=clearCanvas, bg='brown', fg='white', font=('helvetica', 15, 'bold')))

practice_makeDataset()

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

wn.place(x=120, y=0)
wn.bind('<B1-Motion>', paint)
background.pack()

# root.attributes('-fullscreen', True)
root.mainloop()


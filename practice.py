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

currentLetter='A'
dotSize = 5
threshold = 15
index = 0
strokeIndex = 1
strokePointIndex = 0
root = Tk()
root.title("Practice")
root.geometry("970x700")
letter = []
xshift = 0
yshift = 0
CurrentSizeD = 1
dataset=[]
started = False
initX = 0
initY = 0
nStepAgo = [0,0]
stepsCrossed = 0
angleThresh = 20
say=False
command='Let\'s Start'
letterChoiceV='A'
fontsizeV = 1


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
    global letterChoiceV
    global fontsizeV
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
                    if text.find('write') != -1 or text.find('letter') != -1 or text.find('capital') != -1 or text.find('small') != -1:
                        if text.find('letter') != -1:
                            if text.find('capital') != -1:
                                strv = text.split()
                                str = strv[-1]
                                str = "cap"+str.upper()
                                letterChoiceV = str
                                getDataV()
                            else:
                                strv = text.split()
                                str = strv[-1]
                                letterChoiceV = str
                                getDataV()

                    elif text.find('font') != -1 or text.find('size') != -1 or text.find('set') != -1 :
                        strv = text.split()
                        str = strv[-1]
                        fontsizeV = 0
                        if (str == "zero"):
                            fontsizeV = 0
                        else:
                            fontsizeV = int(str)
                        
                        getSizeV()

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

def makeDataset():
    global dataset
    global letter
    global currentLetter
    global CurrentSizeD
    global index,strokePointIndex,strokeIndex
    global started
    global threshold
    global dotSize
    wn.delete('all')
    index = 0
    strokePointIndex = 0
    strokeIndex = 1
    started = False
    letter=[]
    dataset = []
    filename = "letters/"+currentLetter+".txt"
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
        dataset.append(pointlist)
        
    for stroke in dataset:
        for point in stroke:
            point[0] = point[0]/CurrentSizeD
            point[1] = point[1]/CurrentSizeD
            
        
    letter = dataset[0]
    print("current letter = ",currentLetter)
    print("current size = ",CurrentSizeD)
    
    threshold = threshold/CurrentSizeD
    dotSize = dotSize/CurrentSizeD
       
    # add = 0
    # for letters in dataset:
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
    
def getData(event):
    global letterChoice
    global currentLetter
    
    
    newLetter = letterChoice.get()
    if newLetter != currentLetter:
        currentLetter = newLetter
        makeDataset()
        
def getSize(event):
    global dataset
    global fontsizeC
    global CurrentSizeD
    data = fontsizeC.current()
    divisors = [4,2,(4/3),1,0.8,(2/3),(4/7)]
    newSizeD = divisors[data]
        
    if newSizeD != CurrentSizeD:
        CurrentSizeD = newSizeD
        makeDataset()

def getDataV():
    global letterChoiceV
    global currentLetter
    
    
    newLetter = letterChoiceV
    if newLetter != currentLetter:
        currentLetter = newLetter
        makeDataset()
        
def getSizeV():
    global dataset
    global fontsizeV
    global CurrentSizeD
    data = fontsizeV
    divisors = [4,2,(4/3),1,0.8,(2/3),(4/7)]
    newSizeD = divisors[data]
        
    if newSizeD != CurrentSizeD:
        CurrentSizeD = newSizeD
        makeDataset()

            

        
def clearCanvas():
    makeDataset()


#####proper instructions

def correctionMSGpoint(x,y,tx,ty):
    x1,y1,ovalStartx1,ovalStarty1 = x,y,tx,ty
    x2=x1+threshold
    y2=y1+threshold
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



def responseNext(x,y,tx,ty):
    x1,y1,ovalStartx1,ovalStarty1 = x,y,tx,ty
    ovalEndx1,ovalEndy1 = tx,ty
    x2=x1+threshold
    y2=y1+threshold
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
    global strokeIndex,strokePointIndex,index
    global initX,initY,nStepAgo
    global started
    global dataset
    global command
    global say
    if index == len(letter):
        command='Done'
        say=True
        print("Done!!!")
        return 
    
    x = event.x 
    y = event.y 
    x1, y1 = (x-dotSize), (y-dotSize)
    x2, y2 = (x+dotSize), (y+dotSize)
    color = "black"
    
    
    targetX = 0
    targetY = 0
    if started == False:
        initX = x-dataset[0][index][0]
        initY = y-dataset[0][index][1]
        started = True
        for point in letter:
            xc = point[0]+initX
            yc = point[1]+initY
            if xc>820 :
                print("too much on right side, go left")
                makeDataset()
                break
            if xc<10 :
                print("too much on left side, go right")
                makeDataset()
                break
            if yc<10 :
                print("too much on upperside, go down")
                makeDataset()
                break
            if yc>690 :
                print("too much on bottom side, go up")
                makeDataset()
                break
    
    targetX = dataset[0][index][0]+initX
    targetY = dataset[0][index][1]+initY
    angleNeed = 0
    draw_point = True
    
    
    
    if x>=targetX-threshold and x<= targetX+threshold and y>=targetY-threshold and y<= targetY+threshold:
        xt1=targetX-threshold
        xt2=targetX+threshold
        yt1=targetY-threshold
        yt2=targetY+threshold
        print("found!!!")
        wn.create_oval(xt1, yt1, xt2, yt2, fill='white', outline=color,tags='follow')
        index=index+1
        strokePointIndex = strokePointIndex+1
        if index<len(letter):
            targetX = dataset[0][index][0]+initX
            targetY = dataset[0][index][1]+initY
    
    else:
        pretX,pretY = 0,0
        if index > 0:
            pretX = dataset[0][index-1][0]+initX
            pretY = dataset[0][index-1][1]+initY
            
        if not(x>=pretX-threshold and x<= pretX+threshold and y>=pretY-threshold and y<= pretY+threshold):
            print(correctionMSGpoint(x,y,pretX,pretY))
            draw_point = False
    
    
    if draw_point:
        wn.create_oval(x1, y1, x2, y2, fill=color, outline=color,tags='paint')
        print(responseNext(x,y,targetX,targetY))
        

     
    # addx = initX
    # addy = initY
    
    # for letters in dataset:
    #     for point in letters:
    #         x = point[0]+addx
    #         y = point[1]+addy
    #         wn.create_oval((x-10), (y-10), (x+10), (y+10), fill='yellow', outline='yellow',tags='debug')
    #     break
    

t1= threading.Thread(target=voiceGuide, name='t1')
t1.start()

t2= threading.Thread(target=getAudio, name='t2')
t2.start()
    
background=Canvas(root, width=970, height=700, bg='#eeeeee')
wn=Canvas(root, width=850, height=700, bg='white')








background.create_window(60,  50,window=Button(root,text='Practice', command=gotoPractice, bg='#eeeeee', fg='black', font=('helvetica', 15, 'bold')))
background.create_window(60, 190,window=Button(root,text='Test', command=gotoTest, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 120,window=Button(root,text='Create', command=gotoCreate, bg='brown', fg='white', font=('helvetica', 15, 'bold')))
background.create_window(60, 260,window=Button(root,text='clear', command=clearCanvas, bg='brown', fg='white', font=('helvetica', 15, 'bold')))

makeDataset()

strrng = StringVar() 
label = Label(background, text = "Select the Letter :", font = ("Times New Roman", 9))
label.configure(background='#eeeeee')
label.place(x=5,y=295)
letterChoice = ttk.Combobox(background, width = 10, textvariable = strrng) 

lines = []
with open(r'letters/letterlist.txt') as f:
    for row in f:
        lines.append(row.rstrip('\n'))
letterChoice['values'] = lines
    
letterChoice.bind("<<ComboboxSelected>>",getData)
letterChoice.current(0)
letterChoice.place(x=5, y=320)

strrng = StringVar() 
label = Label(background, text = "Select Font size :", font = ("Times New Roman", 10))
label.configure(background='#eeeeee')
label.place(x=5,y=350)
fontsizeC = ttk.Combobox(background, width = 10, textvariable = strrng) 

lines = ['0(25%)','1(50%)','2(75%)','3(100%)','4(125%)','5(150%)','6(175%)']
fontsizeC['values'] = lines
    
fontsizeC.bind("<<ComboboxSelected>>",getSize)
fontsizeC.current(3)
fontsizeC.place(x=5, y=375)

wn.place(x=120, y=0)
wn.bind('<B1-Motion>', paint)
background.pack()


root.mainloop()


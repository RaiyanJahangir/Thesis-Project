practice_threshold = 15
def practice_responseNext(x,y,tx,ty):
    x1,y1,ovalStartx1,ovalStarty1 = x,y,tx,ty
    ovalEndx1,ovalEndy1 = tx,ty
    x2=x1+practice_threshold
    y2=y1+practice_threshold
    print(" x1 = ",x1," y1 = ",y1," x2 = ",x2," y2 = ",y2," tx = ",tx," ty = ",ty)
    text = ""
    global command
    global say
    if x2 < tx and y2 < ty:
        text = "Go Down Right"
    elif x1 > tx and y2 < ty:
        text = "Go Down Left"
    elif x2 < tx and y1 > ty:
        text = "Go Up Right"
    elif x1 > tx and y1 > ty:
        text = "Go Up Left"
    elif y2 < ty and x1<=tx and x2>tx:
        text = "Go Down"
    elif y1 > ty and x1<=tx and x2>tx:
        text = "Go Up"
    elif x2 < tx and y1<=ty and y2>ty:
        text = "Go Right"
    elif x1 > tx and y1<=ty and y2>ty:
        text = "Go Left"
    
    if len(text) == 0:
        text = "keep on going"

    
    return text


print(practice_responseNext(2,3,5,6))
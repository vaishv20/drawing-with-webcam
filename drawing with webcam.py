from cv2 import CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX
import numpy as np
import cv2
from collections import deque
bluelower = np.array([100,60,60])
blueupper = np.array([140,255,255])
kernel = np.ones((5,5), np.uint8)
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]
bindex = 0
gindex = 0
rindex = 0
yindex = 0
colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorindex = 0
paintwindow = np.zeros((471,636,3)) + 255
paintwindow = cv2.rectangle(paintwindow,(40,1),(140,65),(0,0,0),2)
paintwindow = cv2.rectangle(paintwindow,(160,1),(255,65),colors[0],-1)
paintwindow = cv2.rectangle(paintwindow,(275,1),(370,65),colors[1],-1)
paintwindow = cv2.rectangle(paintwindow,(390,1),(485,65),colors[2],-1)
paintwindow = cv2.rectangle(paintwindow,(505,1),(600,65),colors[3],-1)
cv2.putText(paintwindow,"clear all",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintwindow,"blue",(185,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintwindow,"green",(298,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintwindow,"red",(420,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintwindow,"yellow",(520,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),2,cv2.LINE_AA)
cv2.namedWindow('paint',cv2.WINDOW_AUTOSIZE)
camera = cv2.VideoCapture(0)
while True:
    grabbed,frame = camera.read()
    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    frame = cv2.rectangle(frame,(40,1),(140,65),(0,0,0),2)
    frame = cv2.rectangle(frame,(160,1),(255,65),colors[0],-1)
    frame = cv2.rectangle(frame,(275,1),(370,65),colors[1],-1)
    frame = cv2.rectangle(frame,(390,1),(485,65),colors[2],-1)
    frame = cv2.rectangle(frame,(505,1),(600,65),colors[3],-1)
    cv2.putText(frame,"clear all",(49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
    cv2.putText(frame,"blue",(185,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"green",(298,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"red",(420,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"yellow",(520,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),2,cv2.LINE_AA)
    if not grabbed:
        break
    bluemask = cv2.inRange(hsv, bluelower,blueupper)
    bluemask = cv2.erode(bluemask,kernel,iterations=2)
    bluemask = cv2.morphologyEx(bluemask,cv2.MORPH_OPEN,kernel)
    bluemask = cv2.dilate(bluemask,kernel,iterations=1)
    (cnts, _) = cv2.findContours(bluemask.copy(),cv2.RETR_EXTERNAL,
      cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts)>0:
        cnt = sorted(cnts,key = cv2.contourArea,reverse = True)[0]
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
        m = cv2.moments(cnt)
        center = (int(m['m10']/m['m00']),int(m['m01']/m['m00']))
        if center[1]<= 65:
            if 40<= center[0] <= 140:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]
                bindex = 0
                gindex = 0
                rindex = 0
                yindex = 0      
                paintwindow[67:,:,:] = 255
                frame[67:,:,:] = 255
            elif 160<= center[0] <= 255:
                colorindex = 0
            elif 255 <= center[0] <= 370:
                colorindex = 1
            elif 390 <= center[0] <= 485:
                colorindex = 2
            elif 505 <= center[0] <= 600:
                colorindex = 3
        else:
            if colorindex == 0:
                bpoints[bindex].appendleft(center)
            elif colorindex == 1:
                gpoints[gindex].appendleft(center)
            elif colorindex == 2:
                rpoints[rindex].appendleft(center)
            elif colorindex == 3:
                ypoints[yindex].appendleft(center)          
    else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
        gpoints.append(deque(maxlen=512))
        gindex += 1
        rpoints.append(deque(maxlen=512))
        rindex += 1
        ypoints.append(deque(maxlen=512))
        yindex += 1
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1,len(points[i][j])):
                if points[i][j][k-1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame,points[i][j][k-1],points[i][j][k],colors[i],2)
                cv2.line(paintwindow,points[i][j][k-1],points[i][j][k],colors[i],2)
    cv2.imshow('paint', paintwindow)
    cv2.imshow('camera',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()   
cv2.destroyAllWindows()

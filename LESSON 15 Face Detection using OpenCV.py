import numpy as np
import cv2
print("CV2 version:", cv2.__version__, "run successfully!!")
import time

fps=30
timeStamp=time.time()
# print(timeStamp)
width = 640
height = 360
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# ret, frame = cap.read()
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

faceCascade = cv2.CascadeClassifier('C:\\Users\\Odai\\Documents\\Programirung\\Python Projects\\OpenCV\\haar\\haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('C:\\Users\\Odai\\Documents\\Programirung\\Python Projects\\OpenCV\\haar\\haarcascade_eye.xml')
catCascade = cv2.CascadeClassifier('C:\\Users\\Odai\\Documents\\Programirung\\Python Projects\\OpenCV\\haar\\haarcascade_frontalcatface_extended.xml')

# Detecting colors (Hsv Color Space) - Opencv with Python
# https://youtu.be/Q0IPYlIK-4A

while(True):
    # Capture frame-by-frame
    ret, frame = cam.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)
    faces = faceCascade.detectMultiScale(frameGray,scaleFactor=1.3,
	minNeighbors=10, minSize=(75, 75))
    cats = catCascade.detectMultiScale(frameGray,1.3,5)
    
    if len(faces) == 0:
        print("No faces found")
        cv2.putText(frame, "No faces found", (20,20), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    else:
        print("Number of faces detected: " + str(faces.shape[0]))
        cv2.putText(frame, "Number of faces detected: "+ str(faces.shape[0]), (20,20), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        for face in faces:
            x,y,w,h=face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=1)
            # print('x=',x,'y=',y,'width=',w,'height=',h)
            
            frameROI = frame[y:y+h,x:x+w]
            frameROIGray=cv2.cvtColor(frameROI,cv2.COLOR_BGR2GRAY)
            eyes = eyeCascade.detectMultiScale(frameROIGray,scaleFactor=1.3,
        minNeighbors=10, minSize=(10, 10))
            for eye in eyes:
                xeye,yeye,weye,heye=eye
                cv2.rectangle(frame2[y:y+h,x:x+w],(xeye,yeye),(xeye+weye,yeye+heye),(0,255,255),1)
    
    for cat in cats:
        xcat,ycat,wcat,hcat=cat
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),thickness=1)
        # loop over the cat faces and draw a rectangle surrounding each
        for (i, (x, y, w, h)) in enumerate(cats):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Cat #{}".format(i + 1), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        # print('x=',x,'y=',y,'width=',w,'height=',h)
        
    loopTime=time.time()-timeStamp
    timeStamp=time.time()
    fpsNew=1/loopTime
    fps=.9*fps+.1*fpsNew
    fps=int(fps)
    cv2.putText(frame, str(fps)+' fps', (width-50,height-10), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    

    # Display the resulting frame
    puttext1 = cv2.putText(frame,("Modar Soos in BGR Color"),(10,height-10),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.imshow('Original Frame',frame)
    # Change the position of window after imshow() always
    cv2.moveWindow('Original Frame',0,0)
    
    cv2.imshow('Detect Eyes',frame2)
    # Change the position of window after imshow() always
    cv2.moveWindow('Detect Eyes',640,0)
    
    # cv2.imshow('ROI',frameROI)
    # cv2.moveWindow('ROI',0,height+25)

    # cv2.imshow('ROI EYES',frameROIGray)
    # cv2.moveWindow('ROI EYES',width,height+25)
    
    # Display the resulting frame
    # puttext2 = cv2.putText(gray,("Modar Soos in Gray Color"),(10,350),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    # cv2.imshow('gray1',gray)
    # Change the position of window after imshow() always
    # cv2.moveWindow('gray1',640,0)

    # Display the resulting frame
    # cv2.imshow('gray2',gray)
    # Change the position of window after imshow() always
    # cv2.moveWindow('gray2',0,390)

    # cv2.imshow('original2',frame)
    # Change the position of window after imshow() always
    # cv2.moveWindow('original2',640,390)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cam.release()
cv2.destroyAllWindows()

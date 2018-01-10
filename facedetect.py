import cv2
import numpy as np
# implementation of a simple viola jones algorithm,the xml files provided is a default
#dataset that has good features which are being used and hence classification is possible.
#the xml file is provided to the opecv opensource library  by intel corp.Multiple face can be detected using this approach.
#implemented for a face that is assumed for frontal face only i.e only the front of the face is detected,sideview will require more datasets.

forface = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #exporting the xml files that contain the features onto the program(also loads the cascade classifier) 
foreyes = cv2.CascadeClassifier('haarcascade_eye.xml')                 #it is thus important for the xml files to be
                                                                       #present in the same working directory as this program.
                                                                      
cap =cv2.VideoCapture(0) #declaring the camera object allowing it to capture video 0 is default for onboard webcam.

while True:                                                 #for video sequence it keeps looping
    ret, img = cap.read()                                   #reading the image and returning it to img
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)             #converting it to grayscale(simpler to work with thee haar features are just black and white hence.)
    face = forface.detectMultiScale(gray,1.3,5)             #returns a list of the detected images
    for (x,y,w,h) in face:                                  #traversing through the list 'face' and plotting a rectangle for each positively detected face in red.
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h,x:x+h]                        #inorder to detect eyes we assume eyes only exist in the face and hence limit the region of interest to the face
        roi_color = img[y:y+h,x:x+h]
        eye= foreyes.detectMultiScale(roi_gray)         #thus returns a list of the detected eye images
        for(u,v,c,d) in eye:                                #traversing through the eye list to plot the eye in a rectangle of green
            cv2.rectangle(roi_color,(u,v),(u+c,v+d),(0,255,0),2)
    cv2.imshow('vido feed',img)                             #displayin image feed frame by frame
    bob=cv2.waitKey(30) & 0xff
    if bob ==27:                                              #on pressing esc key program breaks out of while true loop
        break
cap.release()  
cv2.destroyAllWindows()                                     #execution of program stops and all windows are closed
            
    
    

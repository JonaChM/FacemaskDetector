import cv2
import serial

#Relay open function
usb_relay = serial.Serial('COM9', 9600, timeout=1) #The COM port has to be selected manually from the device manager.
on_cmd = b'\xA0\x01\x01\xa2'
off_cmd =  b'\xA0\x01\x00\xa1'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml') 
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_mouth.xml')



def detect(gray, frame):
    mouthFound = False
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w] 
        roi_color = frame[y:y + h, x:x + w] 
        smiles = mouth_cascade.detectMultiScale(roi_gray, 1.8, 20) 
        if 'numpy.ndarray' in str(type(smiles)).split("'"): mouthFound = True

        for (sx, sy, sw, sh) in smiles: 
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2) 
    return frame, mouthFound

#Opens the camera only
video_capture = cv2.VideoCapture(0)

#faceON is the parameter use to control how long the face has to be identified for the program to turn on the solenoide.
faceON = 0 
mouthONRT = bool

#Main program is here
while video_capture.isOpened(): 
   # Captures video_capture frame by frame 
    _, frame = video_capture.read()  
  
    # To capture image in monochrome                     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
  
    # calls the detect() function returns the frame with rectangles and if the mouth has been detected.     
    canvas , mouthON = detect(gray, frame)    
    
    #Keeps memory about if the mouth has ever been detected.
    if mouthON == True:
        mouthONRT = True

    #This section determines the windows for the solenoide to be activated
    if 0 in frame: faceON += 1
    else: faceON -=1
    #Upper and lower windows
    if faceON > 30: faceON = 15
    if faceON < -30: faceON = -10
    #Turns on the solenoide only if the mouth has been detected while the faces has been detected for more than 5 seconds.
    if faceON > 10 and mouthONRT == True:  usb_relay.write(on_cmd) 
    if faceON <= -10: 
        #turns it off if hasnt detected the face for more than 5 seconds and returns the memory of mouth to False
        usb_relay.write(off_cmd)
        mouthONRT = False

    #Control print for validation
    print(faceON, mouthONRT)

    # Displays the result on camera feed                      
    cv2.imshow('Video', canvas)  

    # The control breaks once q key is pressed                         
    if cv2.waitKey(1) & 0xff == ord('q'):                
        break
  
# Release the capture once all the processing is done. 
video_capture.release()                                  
cv2.destroyAllWindows() 
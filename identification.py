import cv2
import pickle
import cv2
import pickle
import datetime
import json
import time

face_cascade = cv2.CascadeClassifier("C:\Program Files\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person name":1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf <= 75:
            time1 =  datetime.datetime.now().strftime("%H:%M:%S")
            
            print(labels[id_],time1,id_)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            NAME = labels[id_]
            color = (0, 0, 255)
            stroke = 2
            cv2.putText(frame, NAME, (x,y), font, 1, color, stroke,cv2.LINE_AA)
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(frame, (x,y), (width, height), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#imports
from ultralytics import YOLO
import cv2
import numpy as np
import math 

#path to custom model
model = YOLO("yolov8Model.pt")

#classes
classNames = ["People"]

#setting up the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#processing the frame and getting the bounding boxes for each face seen
def detect(frame):
    results = model(frame, stream=True)

    faces = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))

    return faces

while True:
    #read the camera stream
    result, video_frame = cap.read()
    #if it isn't successful, stop the program
    if result is False:
        break

    #process a frame
    faces = detect(video_frame)

    #draw a bounding box arounded the detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

    #display the processed stream
    cv2.imshow("You !!", video_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
    #if there's more than one face, display a happy face
    if len(faces) > 0:
        cv2.imshow("Happy/Sad", cv2.imread("images/smile.jpg"))
    #if there isn't, display a sad face
    else:
        cv2.imshow("Happy/Sad", cv2.imread("images/sad.jpg"))

cap.release()
cv2.destroyAllWindows()

#imports
import cv2
import numpy as np

#pretrained model
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#setup camera
video_capture = cv2.VideoCapture(0)

#processing the frame, getting bounding box coordinates
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:

    # read frames from the video
    result, video_frame = video_capture.read()

    #if it isn't successful, stop the program
    if result is False:
        break
    faces = detect_bounding_box(
        video_frame
    )

    #display the processed stream
    cv2.imshow(
        "You !!", video_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    #if there's more than one face, display a happy face
    if len(faces) > 0:
        cv2.imshow("Happy/Sad", cv2.imread("images/smile.jpg"))
    #if there isn't, display a sad face
    else:
        cv2.imshow("Happy/Sad", cv2.imread("images/sad.jpg"))

video_capture.release()
cv2.destroyAllWindows()
# AI-Face-Detection
A small program that displays a happy face if it sees you and a sad face if it doesn't using AI

This was a comp eng assignment !

<p align = "center">
  <img src="/images/sad.jpg" width="250" />  
  <img src="/images/smile.jpg" width="250" /> 
</p>

## Custom Detection:
- contains a YOLOv8 model I trained myself on people
- uses OpenCV to show the processed video frame and happy/sad face popup window

## OpenCV Detection:
- uses OpenCV's pretrained model for face detection
- again uses OpenCV to show the processed video frame and happy/sad face popup window

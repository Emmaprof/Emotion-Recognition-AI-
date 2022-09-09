# importing the Libraries
import cv2 as cv
from facial_emotion_recognition import EmotionRecognition

# initializing the EmotionRecognition with CPU
emotion_recognition =  EmotionRecognition(device = "cpu")

# initialize and read the camera
cam = cv.VideoCapture(0)
while True:
    _, frame = cam.read()
    frame = emotion_recognition.recognise_emotion(frame, return_type = "BGR")

    #SHOW WINDOM
    cv.imshow("emotion recognition", frame)
    # close window with escape key
    key = cv.waitKey(1)
    if key == 27:
        break

cam.release()
cv.destroyAllWindows()

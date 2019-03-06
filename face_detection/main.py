import cv2

import logging as log
import datetime as dt
from time import sleep
from face_detection.face_detectors import HaarFaceDetector, DlibFaceDetector

front_cascPath = "haarcascade_frontalface_default.xml"
profile_cascPath = "haarcascade_profileface.xml"


GREEN = (0, 255, 0)
RED = (0, 0, 255)

log.basicConfig(filename='webcam.log', level=log.INFO)


face_detector = HaarFaceDetector(casc_path="face_detection/haar_cascades",
                                 casc_list=[front_cascPath, profile_cascPath])
# face_detector = DlibFaceDetector(model_type="hog")
video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    faces = face_detector.get_face_bbox(frame)
    print(faces)
    face_detector.draw_rectangle(frame, faces)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

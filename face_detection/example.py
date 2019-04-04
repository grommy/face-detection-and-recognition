# python face_detection/example.py

import time

import cv2

import logging as log
import datetime as dt
from time import sleep
from face_detection import HaarFaceDetector, DlibFaceDetector, front_cascPath, \
    profile_cascPath, DnnDetector

GREEN = (0, 255, 0)
RED = (0, 0, 255)

log.basicConfig(filename='webcam.log', level=log.INFO)


# face_detector = HaarFaceDetector(casc_path="face_detection/haar_cascades",
#                                  casc_list=[front_cascPath, profile_cascPath])
# face_detector = DlibFaceDetector(model_type="hog")
face_detector = DnnDetector(model_path="face_detection/detection_models", model_type="TF")
video_capture = cv2.VideoCapture(0)

previous = 0
counter = 0
frame_count = 0
tt_execute = 0
while True:

    # if counter % 5 != 0:
    #     continue

    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame_count += 1
    t = time.time()

    faces = face_detector.get_face_bbox(frame)
    face_detector.draw_rectangle(frame, faces)

    tt_execute += time.time() - t
    fps = frame_count / tt_execute
    label = "OpenCV DNN ; FPS : {:.2f}".format(fps)

    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                cv2.LINE_AA)

    if previous != len(faces):
        previous = len(faces)
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

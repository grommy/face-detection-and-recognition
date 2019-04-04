"""
USAGE

python recognize_faces.py \
--detection 'dnn' \
--encodings data/encodings/face_vec.pkl \
--input 'live' \
--display 1


python recognize_faces.py \
--detection 'dnn' \
--encodings data/encodings/face_vec.pkl \
--input data/videos/input/kyiv_ds_test_1.mov \
--display 1 \
--output data/videos/output/kyiv_ds_1.mov
"""


import argparse
import pickle
import sys
import time
from datetime import datetime
import logging as log

import cv2

from imutils.video import VideoStream
from face_detection import DlibFaceDetector, HaarFaceDetector, front_cascPath, profile_cascPath, \
    DnnDetector
from recognition import DlibFaceRecognizer

# construct the argument parser and parse the arguments
from video_tools import FaceVideoStream

log.basicConfig(filename='recognition.log', level=log.INFO)

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input video")
ap.add_argument("-o", "--output", type=str,
                help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
                help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection", type=str, default="dnn",
                help="you can select detection model from `hog`, `dnn`, `haar` ")

args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
known_encodings, known_names = data["encodings"], data["names"]

# detector
if args["detection"] == "dnn":
    detector = DnnDetector(model_path="face_detection/detection_models", model_type="TF")
elif args["detection"] == "hog":
    detector = DlibFaceDetector(model_type="hog")
elif args["detection"] == "haar":
    detector = HaarFaceDetector(casc_path="face_detection/detection_models/haar_cascades",
                                casc_list=[front_cascPath])
else:
    sys.exit()

# recognizer
recognizer = DlibFaceRecognizer(known_encodings, known_names, detector)

# input
if args["input"].upper() == "LIVE":
    # initialize the video stream and pointer to output video file, then
    # allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    video_stream = FaceVideoStream.from_live(vs)
else:
    # initialize the pointer to the video file and the video writer
    video_stream = FaceVideoStream.from_video_file(args["input"])

writer = None

previous = set()
already_seen_persons = set()

frame_count = 0
tt_execute = 0
# loop over frames from the video file stream
while True:
    # grab the next frame
    # (grabbed, frame) = stream.read()
    frame = video_stream.get_frame()
    if frame is None:
        break

    frame_count += 1
    t = time.time()

    # detect the (x, y)-coordinates of the bounding boxes
    boxes = recognizer.detect_faces(frame)
    # compute the facial encodings for each face
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = recognizer.faces_to_vecs(rgb, boxes)
    # get names
    names = recognizer.recognize_faces(encodings)
    recognizer.mark_face(frame, boxes, names)

    tt_execute += time.time() - t
    fps = frame_count / tt_execute
    label = "Detector {} ; FPS : {:.2f}".format(args["detection"].upper(), fps)
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3,
                cv2.LINE_AA)

    # greetings code
    if set(previous) != set(names):
        new_faces = set(names) - already_seen_persons
        previous = set(names)
        if len(new_faces) > 0:
            greeting = "{} Hello {} ".format(str(datetime.now()), ", ".join(new_faces))
            log.info(greeting)
            print(greeting)
            already_seen_persons.update(new_faces)

    # write to file
    if args["output"]:
        video_stream.write_frame(frame, args["output"])

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# close the video file pointers
video_stream.release()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()

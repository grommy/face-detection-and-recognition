"""
USAGE

python encode_faces.py --dataset data/photos/ -e data/encodings/test.pkl -d 'dnn'
"""


# import the necessary packages
import sys

from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
from face_detection import HaarFaceDetector, front_cascPath, DlibFaceDetector, DnnDetector

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection", type=str, default="dnn",
                help="face detection model to use: `haar`, `hog` or `dnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []


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


# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    print("[INFO] processing image {}/{} ({})".format(i + 1,
                                                      len(imagePaths), name))

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    boxes = detector.get_face_bbox(image)
    if len(boxes) < 1:
        continue

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()

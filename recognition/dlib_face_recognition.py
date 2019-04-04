import cv2

import face_recognition
import imutils

from face_detection import GREEN


class DlibFaceRecognizer(object):
    def __init__(self, known_encodings, known_names, face_detector):
        self.face_detector = face_detector
        self.known_encodings = known_encodings
        self.known_names = known_names
        self.ratio = 1

    def faces_to_vecs(self, frame, boxes):
        resized_frame = self._resize_frame(frame)
        rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_encodings(rgb, boxes)

    def _resize_frame(self, frame, resize_width=750):
        # convert the input frame from BGR to RGB then resize it to have
        # a width of 640px (to speedup processing)
        rgb = imutils.resize(frame, width=resize_width)
        self.ratio = frame.shape[1] / float(rgb.shape[1])
        return rgb

    def detect_faces(self, frame):
        resized_frame = self._resize_frame(frame)
        return self.face_detector.get_face_bbox(resized_frame)

    def recognize_faces(self, encodings):
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.known_encodings,
                                                     encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.known_names[i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        return names

    def mark_face(self, frame, boxes, names, color=GREEN, line_width=2):
        for (box, name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(box.top * self.ratio)
            right = int(box.right * self.ratio)
            bottom = int(box.bottom * self.ratio)
            left = int(box.left * self.ratio)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          color, thickness=line_width)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, thickness=line_width)
        return frame

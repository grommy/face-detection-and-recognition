from collections import namedtuple
import cv2

import face_recognition
import os

Box = namedtuple('Box', ['top', 'right', 'bottom', 'left'])

GREEN = (0, 255, 0)
RED = (0, 0, 255)

front_cascPath = "haarcascade_frontalface_default.xml"
profile_cascPath = "haarcascade_profileface.xml"


class FaceDetector(object):
    def get_face_bbox(self, *args, **kwargs):
        pass

    @staticmethod
    def draw_rectangle(frame, bbox_list, color=GREEN, line_width=2):
        for (top, right, bottom, left) in bbox_list:
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness=line_width)

    def get_image(self):
        pass

    @staticmethod
    def __overlap1D(xmin1, xmax1, xmin2, xmax2):
        return (xmax1 > xmin2) and (xmax2 > xmin1)

    def check_bbox_overlap(self, box1, box2):
        assert isinstance(box1, Box) and isinstance(box2, Box)

        are_overlapping = self.__overlap1D(box1.left, box1.right, box2.left, box2.right) and \
            self.__overlap1D(box1.bottom, box1.top, box2.bottom, box2.top)
        return are_overlapping


class HaarFaceDetector(FaceDetector):
    """
    Detects faces using Haar cascades
    """

    def __init__(self, casc_path, casc_list):

        self.cascades = []
        for casc in casc_list:
            full_casc_path = os.path.join(casc_path, casc)  # abs path should be
            print(full_casc_path)
            self.cascades.append(cv2.CascadeClassifier(full_casc_path))

    @staticmethod
    def _prepare_frame(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def get_face_bbox(self, frame):
        gray_frame = self._prepare_frame(frame)

        bbox_list = []
        for cascade in self.cascades:
            faces = cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                # flags=cv2.CASCADE_SCALE_IMAGE
            )
            face_bbox = [Box(y, x + w, y + h, x,) for (x, y, w, h) in faces]

            non_duplicated = []
            for box_new in face_bbox:
                if len(bbox_list) > 0:
                    for box_existing in bbox_list:
                        if self.check_bbox_overlap(box_new, box_existing):
                            break
                    else:
                        non_duplicated.append(box_new)
                else:
                    bbox_list.extend(face_bbox)

        return bbox_list


class DlibFaceDetector(FaceDetector):

    model = None  # "hog"  # cnn

    def __init__(self, model_type):
        self.model = model_type

    @staticmethod
    def _prepare_frame(frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_face_bbox(self, frame):
        # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prepared_frame = self._prepare_frame(frame)
        boxes = face_recognition.face_locations(prepared_frame, model=self.model)
        print(boxes)
        bboxes = [Box(*box) for box in boxes]
        # bboxes = boxes
        return bboxes

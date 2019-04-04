import cv2

from face_detection import Box, GREEN, RED


class FaceVideoStream(object):
    def __init__(self, stream, live=True):
        # self.detector = detector
        self.stream = stream
        self.ratio = 1
        self.writer = None
        self.is_live_stream = live

    @classmethod
    def from_video_file(cls, input_file):
        return cls(cv2.VideoCapture(input_file), live=False)

    @classmethod
    def from_live(cls, stream):
        return cls(stream)

    def release(self):
        self.stream.release()

    def get_frame(self):
        """
        Get frame from the stream
        :return:
        """
        if self.is_live_stream:
            return self.stream.read()
        (grabbed, frame) = self.stream.read()
        if not grabbed:
            # if the frame was not grabbed, then we have reached the
            # end of the stream
            return None
        return frame

    @staticmethod
    def create_writer(output_file_path, shape_1, shape_0):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_file_path, fourcc, 24,
                                 (shape_1, shape_0), True)
        return writer

    def write_frame(self, frame, output_file_path):
        if self.writer is None:
            self.writer = self.create_writer(output_file_path, frame.shape[1], frame.shape[0])

        self.writer.write(frame)

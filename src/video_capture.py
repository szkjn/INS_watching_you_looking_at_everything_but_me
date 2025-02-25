import cv2

class VideoCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

    def is_opened(self):
        return self.cap.isOpened()

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()
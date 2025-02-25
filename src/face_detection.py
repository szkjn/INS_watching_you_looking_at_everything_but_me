import cv2
from src.utils import load_cascade_model

class FaceDetector:
    def __init__(self):
        self.face_cascade = load_cascade_model('haarcascade_frontalface_default.xml')

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
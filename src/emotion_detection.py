import cv2
from fer import FER
from src.utils import load_cascade_model

class EmotionDetector:
    def __init__(self):
        self.emotion_detector = FER()
        self.eye_cascade = load_cascade_model('haarcascade_eye.xml')

    def detect_emotions(self, frame, faces):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes_bounding_boxes = []
        face_emotions = {}

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            emotion, score = self.emotion_detector.top_emotion(face_roi)
            face_emotions[(x, y, w, h)] = emotion.capitalize() if emotion else "Neutral"

            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

            for (ex, ey, ew, eh) in eyes:
                eyes_bounding_boxes.append((x + ex, y + ey, x + ex + ew, y + ey + eh, (x, y, w, h)))

        return eyes_bounding_boxes, face_emotions
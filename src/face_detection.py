import depthai as dai
import cv2
import numpy as np
import time
from src.config import RGB_RESOLUTION, FACE_DETECT_MODEL, CONFIDENCE_THRESHOLD
from src.utils import frameNorm

class FaceDetector:
    def __init__(self):
        self.pipeline = dai.Pipeline()
        self._setup_pipeline()
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.previous_eyes = []  # Store last detected eye positions
        self.last_detection_time = time.time()

    def _setup_pipeline(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setPreviewSize(*RGB_RESOLUTION)
        cam_rgb.setInterleaved(False)
        cam_rgb.setInterleaved(False)

        # Set FPS lower to stabilize detections
        cam_rgb.setFps(15)  # Default is 30; reduce to 15 FPS

        manip = self.pipeline.createImageManip()
        manip.initialConfig.setResize(300, 300)
        manip.setKeepAspectRatio(False)
        cam_rgb.preview.link(manip.inputImage)

        face_nn = self.pipeline.createMobileNetDetectionNetwork()
        face_nn.setBlobPath(FACE_DETECT_MODEL)
        face_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)

        xout_rgb = self.pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")

        xout_nn = self.pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")

        cam_rgb.preview.link(xout_rgb.input)
        manip.out.link(face_nn.input)
        face_nn.out.link(xout_nn.input)

    def get_frame(self, q_rgb):
        in_rgb = q_rgb.tryGet()
        return in_rgb.getCvFrame() if in_rgb is not None else None

    def get_detections(self, q_nn):
        in_nn = q_nn.tryGet()
        return in_nn.detections if in_nn is not None else []

    def process_detections(self, frame, detections):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_eyes = []

        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            face_roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            gray_face = gray_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            eyes = self.eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=7, minSize=(15, 15))

            for (ex, ey, ew, eh) in eyes:
                x1, y1, x2, y2 = bbox[0] + ex, bbox[1] + ey, bbox[0] + ex + ew, bbox[1] + ey + eh
                new_eyes.append((x1, y1, x2, y2))

        # If no eyes detected, use the last valid detections for 0.5s before switching to "I C U"
        if not new_eyes:
            if time.time() - self.last_detection_time < 0.3:
                return self.previous_eyes
            else:
                self.previous_eyes = []  # Clear buffer if no detections for too long
        else:
            self.previous_eyes = new_eyes
            self.last_detection_time = time.time()

        return new_eyes
    
    def _is_duplicate_eye(self, new_eye, prev_eye, threshold=20):
        """Check if the new eye is too close to a previously detected eye."""
        return (
            abs(new_eye[0] - prev_eye[0]) < threshold and
            abs(new_eye[1] - prev_eye[1]) < threshold
        )

import depthai as dai
import cv2
import numpy as np
import time
from src.config import RGB_RESOLUTION, FACE_DETECT_MODEL, CONFIDENCE_THRESHOLD, FPS, EYE_CROP_SCALE_X, EYE_CROP_SCALE_Y
from src.utils import frameNorm

class FaceDetector:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._setup_pipeline()
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.previous_eyes = []  # Store last detected eye positions
        self.last_detection_time = time.time()

    def _setup_pipeline(self):
        cam_rgb = self.pipeline.createColorCamera()
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(2560, 1440)  # 1440p resolution - balanced quality/performance
        cam_rgb.setInterleaved(False)

        # Set FPS for good performance at 1440p
        cam_rgb.setFps(FPS)

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
                # Apply crop scaling to eye bounding box
                center_x = ex + ew // 2
                center_y = ey + eh // 2
                
                # Scale the dimensions
                scaled_width = int(ew * EYE_CROP_SCALE_X)
                scaled_height = int(eh * EYE_CROP_SCALE_Y)
                
                # Calculate new coordinates centered on the original eye
                new_ex = center_x - scaled_width // 2
                new_ey = center_y - scaled_height // 2
                
                # Convert to global frame coordinates
                x1 = max(0, bbox[0] + new_ex)
                y1 = max(0, bbox[1] + new_ey)
                x2 = min(frame.shape[1], bbox[0] + new_ex + scaled_width)
                y2 = min(frame.shape[0], bbox[1] + new_ey + scaled_height)
                
                # Only add if we have a valid region
                if x2 > x1 and y2 > y1:
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

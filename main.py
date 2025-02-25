import cv2
import numpy as np
from fer import FER  # Lightweight facial expression model

# Set screen size to 1920x1080
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Load Haar Cascade models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize FER (Facial Expression Recognition) model
emotion_detector = FER()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for better Haar performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    eyes_bounding_boxes = []
    face_emotions = {}  # Store emotion per face

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y + h, x:x + w]

        # Detect emotion from the face
        emotion, score = emotion_detector.top_emotion(face_roi)

        # Assign emotion or default to "Neutral"
        face_emotions[(x, y, w, h)] = emotion.capitalize() if emotion else "Neutral"

        # Detect eyes within the face region
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            eyes_bounding_boxes.append((x + ex, y + ey, x + ex + ew, y + ey + eh, (x, y, w, h)))

    # Create a blank screen at 1920x1080
    output_screen = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    # Case 1: No Eyes Detected → Show Black Screen with "I C U"
    if len(eyes_bounding_boxes) == 0:
        text = "I C U"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 5
        text_color = (255, 255, 255)

        # Centering the text
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (SCREEN_WIDTH - text_size[0]) // 2
        text_y = (SCREEN_HEIGHT + text_size[1]) // 2

        cv2.putText(output_screen, text, (text_x, text_y), font, font_scale, text_color, font_thickness)
        cv2.imshow('Eye Detection', output_screen)

    # Case 2: Eyes Detected → Dynamically Adjust Layout
    else:
        num_eyes = len(eyes_bounding_boxes)

        # Determine grid layout based on the number of eyes
        if num_eyes == 1:
            rows, cols = 1, 1  # Full screen for one eye
        elif num_eyes == 3:
            rows, cols = 3, 1  # 3 vertical splits for three eyes
        else:
            rows = int(np.ceil(num_eyes / 2))  # At least 2 eyes per row
            cols = 2  # Always 2 columns for even splits

        # Adjust height based on the number of rows
        split_width = SCREEN_WIDTH // cols
        split_height = SCREEN_HEIGHT // rows

        for i, (x1, y1, x2, y2, face_box) in enumerate(eyes_bounding_boxes):
            # Extract eye region
            eye_img = frame[y1:y2, x1:x2]

            # Resize eye to fit its split section
            resized_eye = cv2.resize(eye_img, (split_width, split_height))

            # Calculate row and column position
            row_idx = i // cols
            col_idx = i % cols

            # Determine placement in output screen
            start_x = col_idx * split_width
            start_y = row_idx * split_height

            output_screen[start_y:start_y + split_height, start_x:start_x + split_width] = resized_eye

            # Retrieve the correct emotion for this eye's face
            emotion_label = face_emotions.get(face_box, "Neutral")

            # Overlay emotion label at bottom-left of the split
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7  # Small font size
            font_thickness = 2
            text_color = (0, 255, 0)  # Green text

            # Calculate bottom-left position for text
            text_x = start_x + 10  # Slight padding from left
            text_y = start_y + split_height - 10  # 10px from the bottom

            cv2.putText(output_screen, emotion_label, (text_x, text_y), font, font_scale, text_color, font_thickness)

        cv2.imshow('Eye Detection', output_screen)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

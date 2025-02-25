from src.video_capture import VideoCapture
from src.face_detection import FaceDetector
from src.emotion_detection import EmotionDetector
from src.display import Display
from src.config import SCREEN_WIDTH, SCREEN_HEIGHT

def main():
    video_capture = VideoCapture()
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    display = Display(SCREEN_WIDTH, SCREEN_HEIGHT)

    while video_capture.is_opened():
        frame = video_capture.read_frame()
        if frame is None:
            break

        faces = face_detector.detect_faces(frame)
        eyes_bounding_boxes, face_emotions = emotion_detector.detect_emotions(frame, faces)

        output_screen = display.create_output_screen(eyes_bounding_boxes, face_emotions, frame)
        display.show_output_screen(output_screen)

        if display.check_exit_condition():
            break

    video_capture.release()
    display.destroy_all_windows()

if __name__ == "__main__":
    main()
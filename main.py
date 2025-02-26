import depthai as dai
from src.face_detection import FaceDetector
from src.display import Display

def main():
    detector = FaceDetector()
    display = Display()

    with dai.Device(detector.pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        q_nn = device.getOutputQueue("nn")

        while True:
            frame = detector.get_frame(q_rgb)
            detections = detector.get_detections(q_nn)

            if frame is not None:
                eyes_bounding_boxes = detector.process_detections(frame, detections)
                
                # Sort eyes left to right to avoid duplication issues
                eyes_bounding_boxes.sort(key=lambda eye: eye[0])

                output_screen = display.create_output_screen(eyes_bounding_boxes, frame)
                display.show_output_screen(output_screen)

            if display.check_exit_condition():
                break
            
if __name__ == "__main__":
    main()

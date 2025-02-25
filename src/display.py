import cv2
import numpy as np
from src.config import DEBUG_MODE, RGB_RESOLUTION

class Display:
    def __init__(self):  # Increased default size for better visibility
        self.width, self.height = RGB_RESOLUTION
        self.fullscreen = False  # Start in windowed mode

    def create_output_screen(self, eyes_bounding_boxes, frame):
        output_screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not eyes_bounding_boxes:
            self._display_no_eyes(output_screen)
        else:
            self._display_eyes(eyes_bounding_boxes, frame, output_screen)

        if DEBUG_MODE:
            debug_screen = self.create_debug_screen(frame, eyes_bounding_boxes)
            self.show_debug_screen(debug_screen)  # Show the debug window separately

        return output_screen  # Only return the eye split screen

    def create_debug_screen(self, frame, eyes_bounding_boxes):
        debug_frame = frame.copy()

        # Draw bounding boxes on the original frame
        for (x1, y1, x2, y2) in eyes_bounding_boxes:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for eyes

        debug_resized = cv2.resize(debug_frame, (self.width, self.height))
        return debug_resized

    def show_debug_screen(self, debug_screen):
        """Show the original camera feed with bounding boxes in a separate window."""
        cv2.namedWindow('Debug View - Full Frame', cv2.WINDOW_NORMAL)
        self._apply_window_state('Debug View - Full Frame')
        cv2.imshow('Debug View - Full Frame', debug_screen)

    def _display_no_eyes(self, output_screen):
        text = "I C U"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 5
        text_color = (255, 255, 255)

        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (self.width - text_size[0]) // 2
        text_y = (self.height + text_size[1]) // 2

        cv2.putText(output_screen, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    def _display_eyes(self, eyes_bounding_boxes, frame, output_screen):
        num_eyes = len(eyes_bounding_boxes)
        rows, cols = self._determine_grid_layout(num_eyes)
        split_width = self.width // cols
        split_height = self.height // rows

        for i, (x1, y1, x2, y2) in enumerate(eyes_bounding_boxes):
            eye_img = frame[y1:y2, x1:x2]
            if eye_img.size == 0:
                continue  # Skip invalid bounding boxes
            resized_eye = cv2.resize(eye_img, (split_width, split_height))

            row_idx = i // cols
            col_idx = i % cols
            start_x = col_idx * split_width
            start_y = row_idx * split_height

            output_screen[start_y:start_y + split_height, start_x:start_x + split_width] = resized_eye

    def _determine_grid_layout(self, num_eyes):
        if num_eyes == 1:
            return 1, 1
        elif num_eyes == 3:
            return 3, 1
        else:
            rows = int(np.ceil(num_eyes / 2))
            cols = 2
            return rows, cols

    def show_output_screen(self, output_screen):
        """Show the processed eye detection output in a separate window and resize it to fit more space on screen."""
        cv2.namedWindow('Eye Detection', cv2.WINDOW_NORMAL)
        self._apply_window_state('Eye Detection')
        cv2.imshow('Eye Detection', output_screen)

    def _apply_window_state(self, window_name):
        if self.fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            self.width, self.height = RGB_RESOLUTION
            cv2.resizeWindow(window_name, self.width, self.height)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    def check_exit_condition(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            self.fullscreen = not self.fullscreen  # Toggle fullscreen mode
        return key == ord('q')

    def destroy_all_windows(self):
        cv2.destroyAllWindows()

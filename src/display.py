import cv2
import numpy as np
from src.config import DEBUG_MODE

class Display:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

    def create_output_screen(self, eyes_bounding_boxes, frame):
        output_screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not eyes_bounding_boxes:
            self._display_no_eyes(output_screen)
        else:
            self._display_eyes(eyes_bounding_boxes, frame, output_screen)

        if DEBUG_MODE:
            debug_screen = self.create_debug_screen(frame, eyes_bounding_boxes)
            return self._combine_debug_view(output_screen, debug_screen)

        return output_screen

    def create_debug_screen(self, frame, eyes_bounding_boxes):
        debug_frame = frame.copy()

        # Draw bounding boxes on the original frame
        for (x1, y1, x2, y2) in eyes_bounding_boxes:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for eyes

        debug_resized = cv2.resize(debug_frame, (self.width, self.height))
        return debug_resized

    def _combine_debug_view(self, output_screen, debug_screen):
        # Stack the debug feed and eye display side by side
        combined_display = np.hstack((debug_screen, output_screen))
        return combined_display
    
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
                continue  # Skip if bounding box is out of frame bounds
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
        cv2.imshow('Eye Detection', output_screen)

    def check_exit_condition(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

    def destroy_all_windows(self):
        cv2.destroyAllWindows()

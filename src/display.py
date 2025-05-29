import cv2
import numpy as np
import time
import random
from src.config import (
    RGB_RESOLUTION, 
    MAIN_TEXT_FONT_SCALE, 
    MAIN_TEXT_FONT_WEIGHT, 
    MAIN_TEXT_VERTICAL_OFFSET, 
    AVAILABLE_FONTS, 
    FONT_NAMES, 
    GRID_ROWS, 
    GRID_COLS, 
    DISPLAY_MODE
)

class Display:
    def __init__(self): 
        self.width, self.height = RGB_RESOLUTION
        self.fullscreen = False
        self.color = True
        self.vertical_flip = False
        self.current_font_index = 0  # Track current font index
        self.display_mode = DISPLAY_MODE  # Track current display mode
        
        # Eye tracking for PARSE_GRID mode
        self.tracked_eyes = {}  # {eye_id: {'bbox': (x1,y1,x2,y2), 'grid_pos': (row,col)}}
        self.next_eye_id = 0
        self.used_positions = set()  # Track which grid positions are occupied

    def create_output_screen(self, eyes_bounding_boxes, frame):
        output_screen = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if not self.color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Keep 3 channels for compatibility

        if not eyes_bounding_boxes:
            self._display_no_eyes(output_screen)
        else:
            self._display_eyes(eyes_bounding_boxes, frame, output_screen)

        return output_screen

    def show_output_screen(self, output_screen):
        """Show the processed eye detection output in a separate window and resize it to fit more space on screen."""
        # Apply vertical flip if enabled
        if self.vertical_flip:
            output_screen = cv2.flip(output_screen, 0)
            
        cv2.namedWindow('Eye Detection', cv2.WINDOW_NORMAL)
        
        if self.fullscreen:
            cv2.setWindowProperty('Eye Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('Eye Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Eye Detection', self.width, self.height)
            
        cv2.imshow('Eye Detection', output_screen)

    def check_keyboard_interaction(self, frame):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            self.fullscreen = not self.fullscreen
        if key == ord('c'):
            self.color = not self.color
        if key == ord('r'):
            self.vertical_flip = not self.vertical_flip
        if key == ord('a'):
            # Cycle to next font
            self.current_font_index = (self.current_font_index + 1) % len(AVAILABLE_FONTS)
            print(f"Font changed to: {FONT_NAMES[self.current_font_index]}")
        if key == ord('w'):
            # Switch display mode between FULL_GRID and PARSE modes
            if self.display_mode == "FULL_GRID":
                self.display_mode = "PARSE_MODE_X3"
            else:
                self.display_mode = "FULL_GRID"
            print(f"Display mode changed to: {self.display_mode}")
        if key == ord('x'):
            # Toggle between PARSE modes
            if self.display_mode == "PARSE_MODE_X3":
                self.display_mode = "PARSE_MODE_X2"
            elif self.display_mode == "PARSE_MODE_X2":
                self.display_mode = "PARSE_MODE_X3"
            else:
                self.display_mode = "PARSE_MODE_X2"  # Switch from FULL_GRID to X2
            print(f"Parse mode changed to: {self.display_mode}")
        if key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snap_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"Image saved: {filename}")
        if key == ord('q'):
            return True

    def check_exit_condition(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
    
    def destroy_all_windows(self):
        cv2.destroyAllWindows()

    def _display_no_eyes(self, output_screen):
        lines = [
            "WATCHING YOU LOOKING",
            "AT EVERYTHING BUT ME",
        ]
        
        font = AVAILABLE_FONTS[self.current_font_index]  # Use current font from cycling
        font_scale = MAIN_TEXT_FONT_SCALE
        font_thickness = MAIN_TEXT_FONT_WEIGHT
        text_color = (255, 255, 255)
        line_spacing = 20  # Space between lines
        
        # Calculate the total text block height
        text_sizes = [cv2.getTextSize(line, font, font_scale, font_thickness)[0] for line in lines]
        total_text_height = sum(h for w, h in text_sizes) + (len(lines) - 1) * line_spacing
        
        # Start position for the first line (centered vertically + offset)
        start_y = (self.height - total_text_height) // 2 + MAIN_TEXT_VERTICAL_OFFSET

        for i, line in enumerate(lines):
            text_size = text_sizes[i]
            text_x = (self.width - text_size[0]) // 2  # Centered horizontally
            text_y = start_y + sum(text_sizes[j][1] for j in range(i)) + i * line_spacing  # Move down each line
            
            cv2.putText(output_screen, line, (text_x, text_y), font, font_scale, text_color, font_thickness)

    def _display_eyes(self, eyes_bounding_boxes, frame, output_screen):
        if self.display_mode == "FULL_GRID":
            self._display_eyes_full_grid(eyes_bounding_boxes, frame, output_screen)
        elif self.display_mode == "PARSE_MODE_X3":
            self._display_eyes_parse_grid_x3(eyes_bounding_boxes, frame, output_screen)
        elif self.display_mode == "PARSE_MODE_X2":
            self._display_eyes_parse_grid_x2(eyes_bounding_boxes, frame, output_screen)

    def _display_eyes_full_grid(self, eyes_bounding_boxes, frame, output_screen):
        """Original mode: cycles through detected eyes to fill all grid positions"""
        rows, cols = GRID_ROWS, GRID_COLS
        split_width = self.width // cols
        split_height = self.height // rows

        # Fill the grid with available eyes, cycling through them if needed
        for row in range(rows):
            for col in range(cols):
                grid_index = row * cols + col
                
                if eyes_bounding_boxes:
                    # Cycle through available eyes if we have fewer than total positions
                    eye_index = grid_index % len(eyes_bounding_boxes)
                    x1, y1, x2, y2 = eyes_bounding_boxes[eye_index]
                    eye_img = frame[y1:y2, x1:x2]
                    
                    if eye_img.size > 0:
                        # Resize eye image to fit the grid cell
                        resized_eye = cv2.resize(eye_img, (split_width, split_height))
                        
                        # Calculate position in the grid
                        start_x = col * split_width
                        start_y = row * split_height
                        
                        # Place the eye image in the grid
                        output_screen[start_y:start_y + split_height, start_x:start_x + split_width] = resized_eye

    def _display_eyes_parse_grid_x3(self, eyes_bounding_boxes, frame, output_screen):
        """PARSE_MODE_X3: Variable multiplication based on eye count"""
        return self._display_eyes_parse_grid_common(eyes_bounding_boxes, frame, output_screen, variable_multiplier=True)
    
    def _display_eyes_parse_grid_x2(self, eyes_bounding_boxes, frame, output_screen):
        """PARSE_MODE_X2: All eyes doubled (x2)"""
        return self._display_eyes_parse_grid_common(eyes_bounding_boxes, frame, output_screen, variable_multiplier=False)

    def _display_eyes_parse_grid_common(self, eyes_bounding_boxes, frame, output_screen, variable_multiplier=True):
        """Common logic for both parse modes with configurable multiplier"""
        if not eyes_bounding_boxes:
            # No eyes detected - clear all tracking
            self.tracked_eyes.clear()
            self.used_positions.clear()
            return
            
        rows, cols = GRID_ROWS, GRID_COLS
        total_positions = rows * cols
        split_width = self.width // cols
        split_height = self.height // rows
        
        # Track which current detections match existing tracked eyes
        current_detections = list(eyes_bounding_boxes)
        matched_eye_ids = set()
        
        # Step 1: Match current detections with existing tracked eyes
        for eye_id, tracked_data in list(self.tracked_eyes.items()):
            old_bbox = tracked_data['bbox']
            best_match = None
            best_overlap = 0
            
            for i, new_bbox in enumerate(current_detections):
                overlap = self._calculate_overlap(old_bbox, new_bbox)
                if overlap > best_overlap and overlap > 0.3:  # 30% overlap threshold
                    best_overlap = overlap
                    best_match = i
            
            if best_match is not None:
                # Update the tracked eye with new bbox
                self.tracked_eyes[eye_id]['bbox'] = current_detections[best_match]
                matched_eye_ids.add(eye_id)
                # Remove matched detection from list
                current_detections.pop(best_match)
            else:
                # Eye disappeared - free all its grid positions (including multiplied instances)
                for instance_key in list(self.used_positions):
                    if isinstance(instance_key, tuple) and len(instance_key) == 2:
                        if instance_key[0] == eye_id:  # This position belongs to the disappeared eye
                            self.used_positions.discard(instance_key)
                del self.tracked_eyes[eye_id]
        
        # Step 2: Assign new detections to new tracked eyes
        for new_bbox in current_detections:
            # Create new tracked eye
            self.tracked_eyes[self.next_eye_id] = {
                'bbox': new_bbox,
                'grid_positions': {}  # Will store positions for each multiplied instance
            }
            self.next_eye_id += 1
        
        # Step 3: Determine multiplication factor
        num_eyes = len(self.tracked_eyes)
        if variable_multiplier:
            # PARSE_MODE_X3: Variable multiplication
            if 1 <= num_eyes <= 8:
                multiplier = 3
            elif 9 <= num_eyes <= 13:
                multiplier = 2
            else:
                multiplier = 1
        else:
            # PARSE_MODE_X2: Fixed doubling
            multiplier = 2
        
        # Step 4: Assign grid positions for each eye instance
        for eye_id, tracked_data in self.tracked_eyes.items():
            if 'grid_positions' not in tracked_data:
                tracked_data['grid_positions'] = {}
                
            # Ensure this eye has positions for all its multiplied instances
            for instance in range(multiplier):
                if instance not in tracked_data['grid_positions']:
                    # Find available positions
                    available_positions = []
                    for row in range(rows):
                        for col in range(cols):
                            pos_key = (eye_id, instance)
                            grid_pos = (row, col)
                            # Check if this grid position is free
                            occupied = False
                            for other_eye_id, other_data in self.tracked_eyes.items():
                                if 'grid_positions' in other_data:
                                    for other_instance, other_pos in other_data['grid_positions'].items():
                                        if other_pos == grid_pos:
                                            occupied = True
                                            break
                                if occupied:
                                    break
                            if not occupied:
                                available_positions.append(grid_pos)
                    
                    if available_positions:
                        # Randomly assign position to this eye instance
                        grid_pos = random.choice(available_positions)
                        tracked_data['grid_positions'][instance] = grid_pos
        
        # Step 5: Render all eye instances
        for eye_id, tracked_data in self.tracked_eyes.items():
            x1, y1, x2, y2 = tracked_data['bbox']
            
            for instance, grid_pos in tracked_data.get('grid_positions', {}).items():
                row, col = grid_pos
                
                eye_img = frame[y1:y2, x1:x2]
                
                if eye_img.size > 0:
                    # Resize eye image to fit the grid cell
                    resized_eye = cv2.resize(eye_img, (split_width, split_height))
                    
                    # Calculate position in the grid
                    start_x = col * split_width
                    start_y = row * split_height
                    
                    # Place the eye image in the grid
                    output_screen[start_y:start_y + split_height, start_x:start_x + split_width] = resized_eye

    def _calculate_overlap(self, bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No overlap
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Return intersection over minimum area
        min_area = min(bbox1_area, bbox2_area)
        return intersection_area / min_area if min_area > 0 else 0.0

    def _determine_grid_layout(self, num_eyes):
        # Return configurable grid layout
        return (GRID_ROWS, GRID_COLS, 0)  # (rows, cols, black_splits)


class DebugDisplay:
    def __init__(self, fps=5):
        self.width, self.height = RGB_RESOLUTION
        self.fps = fps
        self.frame_interval = 1.0 / fps  # Time per frame in seconds
        self.last_update_time = time.time()

    def create_debug_screen(self, frame, eyes_bounding_boxes):
        debug_frame = frame.copy()
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR)  # Maintain 3 channels

        # Draw bounding boxes on the debug frame
        for (x1, y1, x2, y2) in eyes_bounding_boxes:
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        debug_resized = cv2.resize(debug_frame, (self.width, self.height))
        return debug_resized

    def show_debug_screen(self, debug_screen):
        """Show the debug screen only if enough time has passed (throttle FPS)."""
        current_time = time.time()
        if current_time - self.last_update_time >= self.frame_interval:
            cv2.namedWindow('Debug View - Full Frame', cv2.WINDOW_NORMAL)
            cv2.imshow('Debug View - Full Frame', debug_screen)
            self.last_update_time = current_time  # Update timestamp

    def overlay_performance_data(self, frame, perf_data):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9  # Debug text font scale (hardcoded since user doesn't care about it)
        font_thickness = 2
        text_color = (0, 0, 255)  # White text
        bg_color = (0, 0, 0)  # Black background
        padding = 10  # Padding around text
        y_offset = 30  # Initial Y position

        # Calculate background rectangle size
        text_lines = [f"{key}: {value}" for key, value in perf_data.items()]
        text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in text_lines]
        
        max_text_width = max(w for w, h in text_sizes) + 2 * padding
        total_text_height = sum(h for w, h in text_sizes) + (len(text_sizes) - 1) * padding + 2 * padding

        # Draw background rectangle
        cv2.rectangle(frame, (5, 5), (5 + max_text_width, 5 + total_text_height), bg_color, -1)

        # Overlay performance text
        for text in text_lines:
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            cv2.putText(frame, text, (10, y_offset), font, font_scale, text_color, font_thickness)
            y_offset += text_size[1] + padding  # Move down for next line

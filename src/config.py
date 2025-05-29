import blobconverter
import cv2


FPS = 15

# 1440p resolution - good balance between quality and performance  
RGB_RESOLUTION = (2560, 1440)  # QHD/2K - middle ground between 1080p and 4K
EDDA_RESOLUTION = (3024, 1964)
# RGB_RESOLUTION = (1920, 1080)  # Full HD 16:9 - most stable
# RGB_RESOLUTION = (3840, 2160)  # 4K UHD - too demanding for device
# RGB_RESOLUTION = (1280, 720)   # HD 16:9

FACE_DETECT_MODEL = blobconverter.from_zoo('face-detection-retail-0004', shaves=6)
CONFIDENCE_THRESHOLD = 0.5

DEBUG_MODE = True

# Grid layout settings
GRID_ROWS = 3       # Number of rows in the eye grid
GRID_COLS = 9       # Number of columns in the eye grid
# Total eye positions = GRID_ROWS × GRID_COLS (currently 3×9 = 27)

# Eye crop settings
EYE_CROP_SCALE_X = 1.2    # Horizontal crop scaling (1.0 = normal, >1.0 = wider, <1.0 = tighter)
EYE_CROP_SCALE_Y = 1.2    # Vertical crop scaling (1.0 = normal, >1.0 = taller, <1.0 = tighter)
# Examples:
# 1.5 = 50% larger crop area
# 0.8 = 20% smaller crop area
# 2.0 = double the crop area

# Main text font settings
MAIN_TEXT_FONT_SCALE = 1.5
MAIN_TEXT_FONT_WEIGHT = 2
MAIN_TEXT_VERTICAL_OFFSET = 25

# Available font types (for cycling with 'a' key)
AVAILABLE_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX
]

# Font names for display (matches order of AVAILABLE_FONTS)
FONT_NAMES = [
    "HERSHEY_SIMPLEX",
    "HERSHEY_PLAIN", 
    "HERSHEY_DUPLEX",
    "HERSHEY_COMPLEX",
    "HERSHEY_TRIPLEX",
    "HERSHEY_COMPLEX_SMALL",
    "HERSHEY_SCRIPT_SIMPLEX",
    "HERSHEY_SCRIPT_COMPLEX"
]

MAIN_TEXT_FONT_TYPE = AVAILABLE_FONTS[0]  # Default to first font

# Available font types:
# cv2.FONT_HERSHEY_SIMPLEX
# cv2.FONT_HERSHEY_PLAIN
# cv2.FONT_HERSHEY_DUPLEX
# cv2.FONT_HERSHEY_COMPLEX
# cv2.FONT_HERSHEY_TRIPLEX
# cv2.FONT_HERSHEY_COMPLEX_SMALL
# cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
# cv2.FONT_HERSHEY_SCRIPT_COMPLEX
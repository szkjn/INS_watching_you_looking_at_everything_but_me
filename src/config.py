import blobconverter


FPS = 15    # Default is 30; reduce to 15 FPS

# 16:9 aspect ratio resolutions for 6x2 grid layout
RGB_RESOLUTION = (1920, 1080)  # Full HD 16:9
# RGB_RESOLUTION = (1280, 720)   # HD 16:9
# RGB_RESOLUTION = (1440, 1080)

FACE_DETECT_MODEL = blobconverter.from_zoo('face-detection-retail-0004', shaves=6)
CONFIDENCE_THRESHOLD = 0.5

DEBUG_MODE = True
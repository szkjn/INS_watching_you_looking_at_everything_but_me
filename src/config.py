import blobconverter

# Camera settings
# RGB_RESOLUTION = (1920, 1080)
RGB_RESOLUTION = (1280, 720)
# RGB_RESOLUTION = (1440, 1080)

FACE_DETECT_MODEL = blobconverter.from_zoo('face-detection-retail-0004', shaves=6)
CONFIDENCE_THRESHOLD = 0.5

DEBUG_MODE = True
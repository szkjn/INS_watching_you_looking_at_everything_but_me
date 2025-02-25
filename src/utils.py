import cv2

def load_cascade_model(model_path):
    return cv2.CascadeClassifier(cv2.data.haarcascades + model_path)
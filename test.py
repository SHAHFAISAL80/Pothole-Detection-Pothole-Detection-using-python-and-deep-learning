from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO(r"C:\Users\Atif Traders\Pictures\deep learning\Pothole-Detection-Pothole-Detection-using-python-and-deep-learning-main\yolov8_pothole weight\y8best.pt")

# Perform inference on a video file
results = model.predict(source=r"C:\Users\Atif Traders\Pictures\deep learning\Pothole-Detection-Pothole-Detection-using-python-and-deep-learning-main\new.mp4", show=True)
print(results)

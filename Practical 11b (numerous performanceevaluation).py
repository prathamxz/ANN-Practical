# ▶ How to Implement
# 	1.	Prepare a local image named image.jpg or change the path.
# 	2.	Install required libraries.
# 	3.	Run the script: python detect_objects.py
# 	4.	The window will show detected objects with bounding boxes.

# pip install ultralytics opencv-python


from ultralytics import YOLO
import cv2

# Load pretrained YOLOv5s model (small version)
model = YOLO("yolov5s.pt")

# Load an image
img_path = "image.jpg"  # Replace with your image path
img = cv2.imread(img_path)

# Run object detection
results = model(img)

# Plot and show results
results[0].plot()
cv2.imshow("Detected Objects", results[0].plot())
cv2.waitKey(0)
cv2.destroyAllWindows()
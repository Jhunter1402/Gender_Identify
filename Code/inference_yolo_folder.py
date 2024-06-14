import os
import cv2
from ultralytics import YOLO


model = YOLO(r"C:\POC\spill_detection_yolov8\runs\detect\train5\weights\best.pt")

# Confidence threshold for detections
confidence_threshold = 0.4

def detect_objects_in_folder(input_folder, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
    
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        results = model.predict(image)

        for result in results:
            for box in result.boxes:
                if box.conf > confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist() 
                    class_id = int(box.cls)
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, model.names[class_id], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

    print("Object detection completed for all images.")

input_folder = r"C:POC\spill_detection_yolov8\test\compressed"
output_folder = r"spill-test\output-Spill detection Images"

detect_objects_in_folder(input_folder, output_folder)

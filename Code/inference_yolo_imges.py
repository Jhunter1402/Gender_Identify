import os
import cv2
import random

from pathlib import Path
from ultralytics import YOLO


YOLO_TEST_IMG = "D:\ML\data\carton.jpg"

class NF_YOLO:

	def __init__(self, model_path, threshold=0.6) -> None:
		self.model = YOLO(model_path)
		self.threshold = threshold
		self.warm_up_models()
	
	def warm_up_models(self):
		self.model([YOLO_TEST_IMG])
	
	def select_sample(self, img, result):
		num_boxes = len(result.boxes)
		if num_boxes == 0:
			return
		selected_item = random.randint(0, num_boxes - 1) if num_boxes > 1 else 0      
		coor = result.boxes.data[selected_item][0:4]
		c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
		overlay = img.copy()
		cv2.rectangle(overlay, c1, c2, (0, 200, 0), -1)
		alpha = 0.4 
		img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
		return img
  
	def predict(self, img, select_sample=False):
		results = self.model.predict([img], conf=self.threshold)
		result = results[0]
		result_img = result.plot()
		if select_sample:
			result_img = self.select_sample(result_img, result)
		count = len(result.boxes)
		if count == 0:
			result_img = img
		result_json = {
			'count': count,
			'img': result_img
		}
		return result_json

if __name__ == "__main__":
	import cv2
	model_path = "D:\ML\models\cartonV3\carton.pt"
	cc = NF_YOLO(model_path)
	test_img = cv2.imread(YOLO_TEST_IMG)
	cc.predict(test_img, True)

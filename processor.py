import cv2
import random
import colorsys
import numpy as np
from model import darknet
from config import class_names

class frame_processor(object):
	def __init__(self, model_path, img_size=(416, 416)):
		self.model_path = model_path
		self.img_size = img_size
		self.darknet = darknet(self.model_path)

	def process(self, img):
		# input the original image, output the detected image
		input_size = img.shape[:2]
		img_cp = self._preprocess(img)
		bboxes, obj_probs, class_probs = self.darknet.forward(img_cp)
		bboxes, scores, class_max_index = self._process_bboxes(bboxes, obj_probs, class_probs, image_shape=input_size)
		return self._draw_detection(img, bboxes, scores, class_max_index, class_names)

	def _preprocess(self, img):
		img_cp = np.copy(img).astype(np.float32)
		img_rgb = cv2.cvtColor(img_cp, cv2.COLOR_BGR2RGB)
		img_resized = cv2.resize(img_rgb, self.img_size)
		img_normalized = img_resized.astype(np.float32) / 225.0
		img_expanded = np.expand_dims(img_normalized, axis=0)
		return img_expanded

	def _process_bboxes(self, bboxes, obj_probs, class_probs, image_shape=(416, 416), threshold=0.5):
		bboxes = np.reshape(bboxes, [-1, 4])

		bboxes[:, 0:1] *= float(image_shape[1])
		bboxes[:, 1:2] *= float(image_shape[0])
		bboxes[:, 2:3] *= float(image_shape[1])
		bboxes[:, 3:4] *= float(image_shape[0])
		bboxes = bboxes.astype(np.int32)
	
		bbox_min_max = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
		bboxes = self._bboxes_cut(bbox_min_max, bboxes)
	
		obj_probs = np.reshape(obj_probs, [-1])
		class_probs = np.reshape(class_probs, [len(obj_probs), -1])
		class_max_index = np.argmax(class_probs, axis=1)
		class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
		scores = obj_probs * class_probs
	
		keep_index = scores > threshold

		class_max_index, scores, bboxes = class_max_index[keep_index], scores[keep_index], bboxes[keep_index]
		class_max_index, scores, bboxes = self._bboxes_sort(class_max_index, scores, bboxes)
		class_max_index, scores, bboxes = self._bboxes_nms(class_max_index, scores, bboxes)
	
		return bboxes, scores, class_max_index
		
	def _bboxes_cut(self, bbox_min_max, bboxes):
		bboxes = np.copy(bboxes)
		bboxes = np.transpose(bboxes)
		bbox_min_max = np.transpose(bbox_min_max)
		bboxes[0] = np.maximum(bboxes[0], bbox_min_max[0])
		bboxes[1] = np.maximum(bboxes[1], bbox_min_max[1])
		bboxes[2] = np.minimum(bboxes[2], bbox_min_max[2])
		bboxes[3] = np.minimum(bboxes[3], bbox_min_max[3])
		bboxes = np.transpose(bboxes)
		return bboxes

	def _bboxes_sort(self, classes, scores, bboxes, top_k=400):
		index = np.argsort(-scores)
		return classes[index][:top_k], scores[index][:top_k], bboxes[index][:top_k]

	def _bboxes_iou(self, bboxes1, bboxes2):
		bboxes1 = np.transpose(bboxes1)
		bboxes2 = np.transpose(bboxes2)
	
		int_ymin = np.maximum(bboxes1[0], bboxes2[0])
		int_xmin = np.maximum(bboxes1[1], bboxes2[1])
		int_ymax = np.minimum(bboxes1[2], bboxes2[2])
		int_xmax = np.minimum(bboxes1[3], bboxes2[3])
	
		int_h = np.maximum(int_ymax - int_ymin, 0.)
		int_w = np.maximum(int_xmax - int_xmin, 0.)
	
		int_vol = int_h * int_w
		vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
		vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
		IOU = int_vol / (vol1 + vol2 - int_vol)
		return IOU

	def _bboxes_nms(self, classes, scores, bboxes, nms_threshold=0.5):
		keep_bboxes = np.ones(scores.shape, dtype=np.bool)
		for i in range(scores.size - 1):
			if keep_bboxes[i]:
				# Computer overlap with bboxes which are following.
				overlap = self._bboxes_iou(bboxes[i], bboxes[(i+1):])
				# Overlap threshold for keeping + checking part of the same class
				keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i + 1):] != classes[i])
				keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)
	
		idxes = np.where(keep_bboxes)
		return classes[idxes], scores[idxes], bboxes[idxes]

	def _draw_detection(self, img, bboxes, scores, class_max_index, class_names, thr=0.3):
		hsv_tuples = [(x/float(len(class_names)), 1., 1.)  for x in range(len(class_names))]
		colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
		colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
		random.seed(10101)  # Fixed seed for consistent colors across runs.
		random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
		random.seed(None)  # Reset seed to default.
		# draw image
		imgcv = np.copy(img)
		h, w, _ = imgcv.shape
		for i, box in enumerate(bboxes):
			if scores[i] < thr:
				continue
			cls_indx = class_max_index[i]
	
			thick = int((h + w) / 300)
			cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),colors[cls_indx], thick)
			mess = '%s: %.3f' % (class_names[cls_indx], scores[i])
			if box[1] < 20:
				text_loc = (box[0] + 2, box[1] + 15)
			else:
				text_loc = (box[0], box[1] - 10)
			cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (255,255,255), thick//3)
		return imgcv

def test():
	test_processor = frame_processor("./pretrained/tl-yolov2.ckpt")
	random_img = np.random.random([800, 500, 3])
	out = test_processor.process(random_img)
	print(out.shape)

if __name__ == '__main__':
	test()

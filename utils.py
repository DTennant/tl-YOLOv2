import random
import colorsys
import cv2
import numpy as np
import tensorflow as tf

def decode(model_output,output_sizes=(13,13),num_class=80,anchors=None):
	H, W = output_sizes
	num_anchors = len(anchors)
	anchors = tf.constant(anchors, dtype=tf.float32)

	detection_result = tf.reshape(model_output, [-1, H * W, num_anchors, num_class + 5])

	xy_offset = tf.nn.sigmoid(detection_result[:, :, :, 0:2])
	wh_offset = tf.exp(detection_result[:, :, :, 2:4])
	obj_probs = tf.nn.sigmoid(detection_result[:, :, :, 4])
	class_probs = tf.nn.softmax(detection_result[:, :, :, 5:])

	height_index = tf.range(H, dtype=tf.float32)
	width_index = tf.range(W, dtype=tf.float32)
	x_cell, y_cell = tf.meshgrid(height_index, width_index)
	x_cell = tf.reshape(x_cell, [1, -1, 1])
	y_cell = tf.reshape(y_cell, [1, -1, 1])

	# decode
	bbox_x = (x_cell + xy_offset[:, :, :, 0]) / W
	bbox_y = (y_cell + xy_offset[:, :, :, 1]) / H
	bbox_w = (anchors[:, 0] * wh_offset[:, :, :, 0]) / W
	bbox_h = (anchors[:, 1] * wh_offset[:, :, :, 1]) / H
	bboxes = tf.stack([bbox_x - bbox_w / 2, bbox_y - bbox_h / 2,
					   bbox_x + bbox_w / 2, bbox_y + bbox_h / 2], axis=3)

	return bboxes, obj_probs, class_probs

def preprocess(image,image_size=(416,416)):
	image_cp = np.copy(image).astype(np.float32)

	image_rgb = cv2.cvtColor(image_cp,cv2.COLOR_BGR2RGB)
	image_resized = cv2.resize(image_rgb,image_size)

	image_normalized = image_resized.astype(np.float32) / 225.0

	image_expanded = np.expand_dims(image_normalized,axis=0)

	return image_expanded

def postprocess(bboxes,obj_probs,class_probs,image_shape=(416,416),threshold=0.5):
	bboxes = np.reshape(bboxes,[-1,4])

	bboxes[:,0:1] *= float(image_shape[1])
	bboxes[:,1:2] *= float(image_shape[0])
	bboxes[:,2:3] *= float(image_shape[1])
	bboxes[:,3:4] *= float(image_shape[0])
	bboxes = bboxes.astype(np.int32)

	bbox_min_max = [0,0,image_shape[1]-1,image_shape[0]-1]
	bboxes = bboxes_cut(bbox_min_max,bboxes)

	obj_probs = np.reshape(obj_probs,[-1])
	class_probs = np.reshape(class_probs,[len(obj_probs),-1])
	class_max_index = np.argmax(class_probs,axis=1)
	class_probs = class_probs[np.arange(len(obj_probs)),class_max_index]
	scores = obj_probs * class_probs

	keep_index = scores > threshold
	class_max_index = class_max_index[keep_index]
	scores = scores[keep_index]
	bboxes = bboxes[keep_index]

	class_max_index,scores,bboxes = bboxes_sort(class_max_index,scores,bboxes)
	class_max_index,scores,bboxes = bboxes_nms(class_max_index,scores,bboxes)

	return bboxes,scores,class_max_index

def draw_detection(im, bboxes, scores, cls_inds, labels, thr=0.3):
	hsv_tuples = [(x/float(len(labels)), 1., 1.)  for x in range(len(labels))]
	colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
	colors = list(
		map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
	random.seed(10101)  # Fixed seed for consistent colors across runs.
	random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
	random.seed(None)  # Reset seed to default.
	# draw image
	imgcv = np.copy(im)
	h, w, _ = imgcv.shape
	for i, box in enumerate(bboxes):
		if scores[i] < thr:
			continue
		cls_indx = cls_inds[i]

		thick = int((h + w) / 300)
		cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),colors[cls_indx], thick)
		mess = '%s: %.3f' % (labels[cls_indx], scores[i])
		if box[1] < 20:
			text_loc = (box[0] + 2, box[1] + 15)
		else:
			text_loc = (box[0], box[1] - 10)
		cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (255,255,255), thick//3)
	return imgcv

def bboxes_cut(bbox_min_max,bboxes):
	bboxes = np.copy(bboxes)
	bboxes = np.transpose(bboxes)
	bbox_min_max = np.transpose(bbox_min_max)
	bboxes[0] = np.maximum(bboxes[0],bbox_min_max[0])
	bboxes[1] = np.maximum(bboxes[1],bbox_min_max[1])
	bboxes[2] = np.minimum(bboxes[2],bbox_min_max[2])
	bboxes[3] = np.minimum(bboxes[3],bbox_min_max[3])
	bboxes = np.transpose(bboxes)
	return bboxes

def bboxes_sort(classes,scores,bboxes,top_k=400):
	index = np.argsort(-scores)
	classes = classes[index][:top_k]
	scores = scores[index][:top_k]
	bboxes = bboxes[index][:top_k]
	return classes,scores,bboxes

def bboxes_iou(bboxes1,bboxes2):
	bboxes1 = np.transpose(bboxes1)
	bboxes2 = np.transpose(bboxes2)

	int_ymin = np.maximum(bboxes1[0], bboxes2[0])
	int_xmin = np.maximum(bboxes1[1], bboxes2[1])
	int_ymax = np.minimum(bboxes1[2], bboxes2[2])
	int_xmax = np.minimum(bboxes1[3], bboxes2[3])

	int_h = np.maximum(int_ymax-int_ymin,0.)
	int_w = np.maximum(int_xmax-int_xmin,0.)

	int_vol = int_h * int_w
	vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
	vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
	IOU = int_vol / (vol1 + vol2 - int_vol)
	return IOU

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5):
	keep_bboxes = np.ones(scores.shape, dtype=np.bool)
	for i in range(scores.size-1):
		if keep_bboxes[i]:
			# Computer overlap with bboxes which are following.
			overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
			# Overlap threshold for keeping + checking part of the same class
			keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
			keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

	idxes = np.where(keep_bboxes)
	return classes[idxes], scores[idxes], bboxes[idxes]
import numpy as np
import cv2
import utils
from config import class_names, anchors
import tensorflow as tf
from model import darknet
import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_img', type=str, help='the image to be processed')
	parser.add_argument('--model_path', type=str, help='the path to the model ckpt')
	parser.add_argument('--output_img', type=str, help='the path to save the output img')
	return parser

def main():
	parser = get_parser()
	args = parser.parse_args()
	input_size = (416, 416)
	image = cv2.imread(args.input_img)
	image_shape = image.shape[:2]
	image_cp = utils.preprocess(image, input_size)

	img_placeholder = tf.placeholder(tf.float32, shape=[1, input_size[0], input_size[1], 3])
	model = darknet(img_placeholder)
	output_size = (input_size[0] // 32, input_size[1] // 32)
	output_decoded = utils.decode(model.outputs, output_sizes=output_size,
									num_class=len(class_names), anchors=anchors)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, args.model_path)
		bboxes, obj_probs, class_probs = sess.run(output_decoded, feed_dict={img_placeholder: image_cp})

	bboxes, scores, class_max_index = utils.postprocess(bboxes, obj_probs, class_probs, 
														image_shape=image_shape)

	output_img = utils.draw_detection(image, bboxes, scores, class_max_index, class_names)
	cv2.imwrite(args.output_img, output_img)
	print('YOLOv2 detection done!!!')

if __name__ == '__main__':
	main()
import cv2
import sys
import argparse
import numpy as np
from time import time
import tensorflow as tf
from model import darknet
from config import class_names, anchors
from processor import frame_processor

import logging 
logging.getLogger("requests").setLevel(logging.ERROR)

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--run_mode', type=str, choices=['video', 'image'], help='the mode will be video or image')
	parser.add_argument('--input_video', type=str, help='the video to be processed, default: capture from the Camera(if available)')
	parser.add_argument('--output_video', type=str, help='the path to save the video, default: play on the screen(not saving)')
	parser.add_argument('--input_img', type=str, help='the image to be processed')
	parser.add_argument('--output_img', type=str, help='the path to save the output img')
	parser.add_argument('--model_path', type=str, help='the path to the model ckpt')
	return parser

def process_img(args, processor):
	img = cv2.imread(args.input_img)
	start_time = time()
	output_img = processor.process(img)
	cv2.imwrite(args.output_img, output_img)
	end_time = time()
	print('YOLOv2 detection done!!!, takes {} seconds'.format(end_time - start_time))

def process_video(args, processor):
	cap = cv2.VideoCapture(args.input_video)
	tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	start_time = time()
	ret, frame = cap.read()
	fps = cap.get(cv2.CAP_PROP_FPS)
	size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	out = cv2.VideoWriter(args.output_video, fourcc, fps, size)
	num_frame = 1
	while ret:
		frame = processor.process(frame)
		out.write(frame)
		print('Frame No.{} of {}'.format(num_frame, tot_frame))
		num_frame += 1
		ret, frame = cap.read()
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	end_time = time()
	print('YOLOv2 detection done!!!, takes {} seconds'.format(end_time - start_time))

def main():
	parser = get_parser()
	args = parser.parse_args()
	processor = frame_processor(args.model_path)
	if args.run_mode == 'image':
		process_img(args, processor)
	else:
		if not args.input_video: args.input_video = 0
		if not args.output_video: sys.exit('Error: You Need to assign an output video')
		process_video(args, processor)

if __name__ == '__main__':
	main()

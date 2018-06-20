import os
import cv2
import sys
import numpy as np
from time import time
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QBasicTimer
from processor import frame_processor
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import QAction, QFileDialog
from PyQt5.QtWidgets import QMainWindow, QTextEdit
from PyQt5.QtWidgets import QProgressBar, QApplication

import logging 
logging.getLogger("requests").setLevel(logging.ERROR)

class YOLOAPP(QMainWindow):
	def __init__(self):
		super().__init__()
		self.initUI()

	def initUI(self):
		self.processed_file = ''
		self.processed_file_type = ''
		self.processor = frame_processor('pretrained/tl-yolov2.ckpt')

		openFile = QAction(QIcon('open.png'), 'Open', self)
		openFile.setShortcut('Ctrl+O')
		openFile.setStatusTip('Open the image file or video file')
		openFile.triggered.connect(self.open_file)

		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(openFile)

		self.pbar = QProgressBar(self)
		self.pbar.setGeometry(30, 40, 350, 50)

		self.btn = QPushButton('Start Processing', self)
		self.btn.resize(self.btn.sizeHint())
		self.btn.move(400, 50)
		self.btn.clicked.connect(self.doAction)

		self.timer = QBasicTimer()
		self.step = 0

		self.statusBar().showMessage('Ready')

		self.setGeometry(300, 300, 600, 120)
		self.setWindowTitle('YOLOv2 GUI')
		self.show()

	def timerEvent(self, e):
		if self.step >= 100:
			self.timer.stop()
			self.btn.setText('Finished')
			return 
		# Do the work
		if self.processed_file_type == 'jpg':
			img = cv2.imread(self.processed_file)
			start_time = time()
			output_img = self.processor.process(img)
			cv2.imwrite('output_img.jpg', output_img)
			end_time = time()
			self.statusBar().showMessage('YOLOv2 detection done!!!, takes {} seconds'.format(end_time - start_time))
			self.step = self.step + 100
			self.pbar.setValue(self.step)	
		else:
			if self.processed_frame == 0: self.start_time = time()
			ret, frame = self.cap.read()
			if not ret:
				self.timer.stop()
				self.btn.setText('Finished')
				end_time = time()
				self.statusBar().showMessage('YOLOv2 detection done!!!, takes {} seconds'.format(end_time - self.start_time))
				return
			frame = self.processor.process(frame)
			self.out.write(frame)
			self.processed_frame += 1
			self.statusBar().showMessage('Frame No.{} of {}'.format(self.processed_frame, self.tot_frame))
			self.step = int((float(self.processed_frame) / float(self.tot_frame)) * 100)
			self.pbar.setValue(self.step)

	def doAction(self):
		if self.processed_file == '':
			# TODO: more advance the error window
			self.statusBar().showMessage('Error: No input files')
			return 
		if self.timer.isActive():
			self.timer.stop()
			self.btn.setText('Start')
		else:
			self.timer.start(100, self)
			self.btn.setText('Processing...')

	def open_file(self):
		fname, _ = QFileDialog.getOpenFileName(self, 'Open', os.getcwd(), 'Image files(*.jpg);;Video files(*.avi)')
		# process the video or image
		ftype = fname.split('.')[-1]
		self.timer.stop()
		self.btn.setText('Start')
		self.step = 0
		self.pbar.setValue(self.step)
		self.processed_file = fname
		self.processed_file_type = 'jpg' if ftype == 'jpg' else 'avi'
		if self.processed_file_type == 'avi':
			self.cap = cv2.VideoCapture(self.processed_file)
			self.tot_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			fps = self.cap.get(cv2.CAP_PROP_FPS)
			size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
					int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
			self.out = cv2.VideoWriter('output_video.avi', fourcc, fps, size)
			self.processed_frame = 0
		return 

def main():
	app = QApplication(sys.argv)
	ex = YOLOAPP()
	sys.exit(app.exec_())

if __name__ == '__main__':
	main()
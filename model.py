import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from config import class_names, anchors
# disable the [TL] Warnings
import sys
class temp_stdout(object):
	def write(s):
		pass
	def flush():
		pass
# disable TensorFlow Warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


lambda_conv = lambda net, shape_, name_: Conv2dLayer(net, act=tf.identity, shape=shape_, padding='VALID', b_init=None, name=name_)
lambda_bn = lambda net, name_: BatchNormLayer(net, act=lambda x: tl.act.lrelu(x, 0.1), is_train=False, name=name_)
lambda_pad = lambda net, name_: PadLayer(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name=name_)
lambda_max_pool = lambda net, name_: MaxPool2d(net, filter_size=(2, 2), strides=(2, 2), padding='VALID', name=name_)

class PassThroughLayer(Layer):
	def __init__(
		self,
		layer = None,
		strides = 2,
		name = 'PassThrough',
	):
		# check layer name (fixed)
		Layer.__init__(self, layer, name=name)

		# the input of this layer is the output of previous layer (fixed)
		self.inputs = layer.outputs

		# print out info (customized)
		print('  [TL] PassThroughLayer {}: strides:{}'.format(name, strides))

		# operation (customized)
		self.outputs = tf.space_to_depth(self.inputs, block_size=strides)

		# update layer (customized)
		self.all_layers = list(layer.all_layers)
		self.all_params = list(layer.all_params)
		self.all_drop = dict(layer.all_drop)
		self.all_layers.extend([self.outputs])

class MyConcatLayer(Layer):
	def __init__(
		self,
		layers,
		concat_dim = -1,
		name = 'MyConcatLayer',
	):
		# check layer name (fixed)
		Layer.__init__(self, layers, name=name)

		# the input of this layer is the output of previous layer (fixed)
		self.inputs = []
		for l in layers:
			self.inputs.append(l.outputs)

		# print out info (customized)
		s = '  [TL] MyConcatLayer {}: '.format(name)
		for l in layers:
			s = s + l.name + ' '
		print(s + ' concat_dim: {}'.format(concat_dim))
		#print('  [TL] MyConcatLayer {}: concat_dim:{}'.format(name, concat_dim))

		# operation (customized)
		self.outputs = tf.concat(self.inputs, axis=concat_dim)

		# update layer (customized)
		self.all_layers = list(layers[0].all_layers)
		self.all_params = list(layers[0].all_params)
		self.all_drop = dict(layers[0].all_drop)
		
		for i in range(1, len(layers)):
			self.all_layers.extend(list(layers[i].all_layers))
			self.all_params.extend(list(layers[i].all_params))
			self.all_drop.update(dict(layers[i].all_drop))
		
		self.all_layers = list_remove_repeat(self.all_layers)
		self.all_params = list_remove_repeat(self.all_params)

		self.all_layers.append(self.outputs)
		#self.all_layers = list(layer.all_layers)
		#self.all_params = list(layer.all_params)
		#self.all_drop = dict(layer.all_drop)
		#self.all_layers.extend([self.outputs])

class darknet(object):
	def __init__(self, model_path, 
				img_placeholder=tf.placeholder(tf.float32, shape=[None, 416, 416, 3]), 
				n_last_channels=425, session=tf.InteractiveSession()):
		self.model_path = model_path
		self.img_placeholder = img_placeholder
		self.n_last_channels = n_last_channels
		self.sess = session
		self.model = self._build_model(self.img_placeholder, self.n_last_channels)
		self.output = self._decode(self.model.outputs, num_class=len(class_names), anchors=anchors)
		#self.bboxes, self.obj_probs, self.class_probs = self.output
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, self.model_path)

	def forward(self, images):
		bboxes, obj_probs, class_probs = self.sess.run(self.output, feed_dict={self.img_placeholder: images})
		return bboxes, obj_probs, class_probs

	def _build_model(self, img_placeholder, n_last_channels):
		stdout_bak = sys.stdout
		sys.stdout = temp_stdout
		net = InputLayer(img_placeholder)
		net = lambda_pad(net, 'pad1')
		# conv1
		net = lambda_conv(net, [3, 3, 3, 32], 'conv1')
		net = lambda_bn(net, 'conv1_bn')
		net = lambda_max_pool(net, 'pool1')
		# conv2
		net = lambda_pad(net, 'pad2')
		net = lambda_conv(net, [3, 3, 32, 64], 'conv2')
		net = lambda_bn(net, 'conv2_bn')
		net = lambda_max_pool(net, 'pool2')
		# conv3
		net = lambda_pad(net, 'pad3')
		net = lambda_conv(net, [3, 3, 64, 128], 'conv3_1')
		net = lambda_bn(net, 'conv3_1_bn')
		net = lambda_conv(net, [1, 1, 128, 64], 'conv3_2')
		net = lambda_bn(net, 'conv3_2_bn')
		net = lambda_pad(net, 'pad4')
		net = lambda_conv(net, [3, 3, 64, 128], 'conv3_3')
		net = lambda_bn(net, 'conv3_3_bn')
		net = lambda_max_pool(net, 'pool3')
		# conv4
		net = lambda_pad(net, 'pad5')
		net = lambda_conv(net, [3, 3, 128, 256], 'conv4_1')
		net = lambda_bn(net, 'conv4_1_bn')
		net = lambda_conv(net, [1, 1, 256, 128], 'conv4_2')
		net = lambda_bn(net, 'conv4_2_bn')
		net = lambda_pad(net, 'pad6')
		net = lambda_conv(net, [3, 3, 128, 256], 'conv4_3')
		net = lambda_bn(net, 'conv4_3_bn')
		net = lambda_max_pool(net, 'pool4')
		# conv5
		net = lambda_pad(net, 'pad7')
		net = lambda_conv(net, [3, 3, 256, 512], 'conv5_1')
		net = lambda_bn(net, 'conv5_1_bn')
		net = lambda_conv(net, [1, 1, 512, 256], 'conv5_2')
		net = lambda_bn(net, 'conv5_2_bn')
		net = lambda_pad(net, 'pad8')
		net = lambda_conv(net, [3, 3, 256, 512], 'conv5_3')
		net = lambda_bn(net, 'conv5_3_bn')
		net = lambda_conv(net, [1, 1, 512, 256], 'conv5_4')
		net = lambda_bn(net, 'conv5_4_bn')
		net = lambda_pad(net, 'pad9')
		net	= lambda_conv(net, [3, 3, 256, 512], 'conv5_5')
		net = lambda_bn(net, 'conv5_5_bn')
		short_cut = net
		net = lambda_max_pool(net, 'pool5')
		# conv6
		net = lambda_pad(net, 'pad10')
		net = lambda_conv(net, [3, 3, 512, 1024], 'conv6_1')
		net = lambda_bn(net, 'conv6_1_bn')
		net = lambda_conv(net, [1, 1, 1024, 512], 'conv6_2')
		net = lambda_bn(net, 'conv6_2_bn')
		net = lambda_pad(net, 'pad11')
		net = lambda_conv(net, [3, 3, 512, 1024], 'conv6_3')
		net = lambda_bn(net, 'conv6_3_bn')
		net = lambda_conv(net, [1, 1, 1024, 512], 'conv6_4')
		net = lambda_bn(net, 'conv6_4_bn')
		net = lambda_pad(net, 'pad12')
		net = lambda_conv(net, [3, 3, 512, 1024], 'conv6_5')
		net = lambda_bn(net, 'conv6_5_bn')
		# conv7
		net = lambda_pad(net, 'pad13')
		net	= lambda_conv(net, [3, 3, 1024, 1024], 'conv7_1')
		net = lambda_bn(net, 'conv7_1_bn')
		net = lambda_pad(net, 'pad14')
		net = lambda_conv(net, [3, 3, 1024, 1024], 'conv7_2')
		net = lambda_bn(net, 'conv7_2_bn')
	
		short_cut = lambda_conv(short_cut, [1, 1, 512, 64], 'conv_shortcut')
		short_cut = lambda_bn(short_cut, 'conv_shortcut_bn')
		# TODO: change PassThroughLayer and MyConcatLayer into proper tensorlayer layers
		short_cut = PassThroughLayer(short_cut, 2)
		net = MyConcatLayer([short_cut, net], concat_dim=-1, name='concat')
		# conv8
		net = lambda_pad(net, 'pad15')
		net = lambda_conv(net, [3, 3, 1280, 1024], 'conv8')
		net = lambda_bn(net, 'conv8_bn')
		# conv_dec
		net = Conv2dLayer(net, act=tf.identity, shape=[1, 1, 1024, 425],
						 padding='VALID', name='conv_dec')
		sys.stdout = stdout_bak
		return net

	def _decode(self, model_output, output_sizes=(13, 13), num_class=80, anchors=None):
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

def test():
	model_darknet = darknet("./pretrained/tl-yolov2.ckpt")
	out = model_darknet.sess.run(model_darknet.model.outputs, feed_dict={model_darknet.img_placeholder: np.random.random([1, 416, 416, 3])})
	print(out.shape) # Expected to be (1, 13, 13, 425)
	out = model_darknet.forward(np.random.random([1, 416, 416, 3]))
	print(out[0].shape) # Expected to be (1, 169, 5, 4)
	print(out[1].shape) # Expected to be (1, 169, 5)
	print(out[2].shape) # Expected to be (1, 169, 5, 80)

if __name__ == '__main__':
	test()
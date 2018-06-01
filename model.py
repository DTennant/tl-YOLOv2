import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

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
#		print('  [TL] MyConcatLayer {}: concat_dim:{}'.format(name, concat_dim))

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

def darknet(images, n_last_channels=425):
	net = InputLayer(images)
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
	net = lambda_conv(net, [3, 3, 256, 512], 'conv5_5')
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
	net = lambda_conv(net, [3, 3, 1024, 1024], 'conv7_1')
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
	return net

def test():
	x = tf.random_normal([1, 416, 416, 3])
	model = darknet(x)
	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, "./pretrained/tl-yolov2.ckpt")
		print(sess.run(model.outputs).shape) # (1,13,13,425)

if __name__ == '__main__':
	test()

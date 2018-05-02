import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim import assign_from_checkpoint_fn
from tensorflow.contrib.slim.nets import vgg
import argparse
import json
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.json')


def read_image(filename, image_size):
	image = cv2.imread(filename)
	image = cv2.resize(image, (image_size, image_size))
	image = tf.cast(image, tf.float32)
	image = tf.expand_dims(image, axis=0)

	return image


if __name__ == '__main__':
	args = parser.parse_args()
	config_path = args.config

	with open(config_path) as f:
		config = json.load(f)

	with open(config['label_path']) as f:
		label = json.load(f)

	graph = tf.Graph()
	with graph.as_default():
		with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=config['weight_decay'])):
			img = read_image(os.path.join('data', 'dog.jpg'), config['image_size'])
			print(img)
			logits, _ = vgg.vgg_16(img, is_training=False)
			prob = tf.nn.softmax(logits)

		# print(slim.get_model_variables('vgg_16'))
		init_fb = assign_from_checkpoint_fn(config['model_path'], slim.get_model_variables('vgg_16'))

		with tf.Session(graph=graph) as sess:
			init_fb(sess)

			predict = sess.run(prob)
			predict = predict[0, 0:]

			print(predict.shape)
			sorted_inds = [i[0] for i in sorted(enumerate(-predict), key=lambda x: x[1])]



	for i in range(5):
		index = sorted_inds[i]
		print('Probability %0.2f => [%s]' % (predict[index], label[str(index)][1]))














import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim import assign_from_checkpoint_fn
from tensorflow.contrib.slim.nets import vgg
import argparse
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.json')


def read_image(filename, image_size=None):
	image = cv2.imread(filename)
	if image_size is not None:
		image = cv2.resize(image, (image_size, image_size))
	image = tf.cast(image, tf.float32)
	image = tf.expand_dims(image, axis=0)

	return image


def discrete_matshow(data, labels_names=[], title=""):
	# get discrete colormap
	cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)
	# set limits .5 outside true range
	mat = plt.matshow(data,
					  cmap=cmap,
					  vmin=np.min(data) - .5,
					  vmax=np.max(data) + .5)
	# tell the colorbar to tick at integers
	cax = plt.colorbar(mat,
					   ticks=np.arange(np.min(data), np.max(data) + 1))

	# The names to be printed aside the colorbar
	if labels_names:
		cax.ax.set_yticklabels(labels_names)

	if title:
		plt.suptitle(title, fontsize=14, fontweight='bold')


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
			# img = read_image(os.path.join('data', 'dog.jpg'))
			image_string = tf.read_file('data/cat.jpg')
			img_decoded = tf.image.decode_jpeg(image_string, channels=3)
			img = tf.cast(img_decoded, tf.float32)
			img = tf.expand_dims(img, axis=0)
			logits, _ = vgg.vgg_16(img, is_training=False, spatial_squeeze=False)

		pred = tf.argmax(logits, dimension=3)
		init_fb = assign_from_checkpoint_fn(config['model_path'], slim.get_model_variables('vgg_16'))

		with tf.Session(graph=graph) as sess:
			init_fb(sess)

			segmentation = sess.run(pred)

		segmentation = np.squeeze(segmentation)
		print(segmentation.shape)
		unique_classes, relabeled_image = np.unique(segmentation,
													return_inverse=True)
		segmentation_size = segmentation.shape
		relabeled_image = relabeled_image.reshape(segmentation_size)
		labels_name = []

		for index, current_class_number in enumerate(unique_classes):
			labels_name.append(str(index) + ' ' + label[str(current_class_number)][1])


		discrete_matshow(data=relabeled_image, labels_names=labels_name, title="Segmentation")











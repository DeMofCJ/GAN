import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp
import tensorflow as tf
import sys, os


flags = tf.app.flags

flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "learning rate for Adam[0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of  train images [np.inf]")
flags.DEFINE_integer("image_size", 108, "The size of image to use")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("output_size", 64, "The size of the output iamges to produce [64]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [mnist, cifar10]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")


FLAGS = flags.FLAGS

def main(_):
	pp.pprint(flags.FLAGS.__flags)

	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)

	with tf.Session() as sess:
		if FLAGS.dataset == 'mnist':
			dcgan = DCGAN(sess, image_size=108, batch_size=64, y_dim=None, output_size=28, c_dim=1,
						dataset_name='mnist', checkpoint_dir=FLAGS.checkpoint_dir, sample_dir=FLAGS.sample_dir)
		else:
			pass

		if FLAGS.is_train:
			dcgan.train(FLAGS)
		else:
			pass


if __name__ == '__main__':
	tf.app.run()





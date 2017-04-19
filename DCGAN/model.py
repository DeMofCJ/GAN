from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import time
import glob
import numpy as np
import tensorflow as tf
from six.moves import xrange
from utils import *
import os,sys


sys.path.append('..')
from loadData.loadData import *


def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak * x)

class DCGAN(object):
	def __init__(self, sess, image_size=64, batch_size=64, sample_size=64, output_size=28,
				y_dim=None, z_dim=100, gf_dim=64, df_dim=64, c_dim=None,
				dataset_name=None, checkpoint_dir=None, sample_dir=None):

		self.sess = sess
		self.image_size = image_size
		self.batch_size = batch_size
		self.output_size = output_size
		self.sample_size = sample_size
		self.y_dim = y_dim
		self.z_dim = z_dim
		self.gf_dim = gf_dim
		self.df_dim = df_dim
		self.c_dim = c_dim

		self.dataset_name = dataset_name
		self.checkpoint_dir = checkpoint_dir
		self.sample_dir = sample_dir
		self.build_model()

	def build_model(self):
		# data
		image_dims = [self.output_size, self.output_size, self.c_dim]
		self.images = tf.placeholder(tf.float32, [self.batch_size]+image_dims, name='real_images')
		self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

		# nets
		self.G = self.generator(self.z)
		self.D, self.D_logits = self.discriminator(self.images, reuse=False)
		self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

		# loss
		self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
		self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
		self.D_loss = self.D_loss_real + self.D_loss_fake
		self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

		# self.D_loss_real = -tf.reduce_mean(self.D_logits)
		# self.D_loss_fake = tf.reduce_mean(self.D_logits_)
		# self.D_loss = self.D_loss_real + self.D_loss_fake
		# self.G_loss = -tf.reduce_mean(self.D_logits_)
		
		t_vars = tf.trainable_variables()
		self.d_vars = [var for var in t_vars if "discriminator" in var.name]
		self.g_vars = [var for var in t_vars if "generator" in var.name]

		self.saver = tf.train.Saver()

	def train(self, config):
		# if config.dataset == 'mnist':
		# 	data_X, _ = self.load_mnist()
		# else:
		# 	pass
		
		data_X, _ = load_mnist('mnist')
		d_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1).minimize(self.D_loss, var_list=self.d_vars)
		g_optim = tf.train.AdamOptimizer(learning_rate=config.learning_rate, beta1=config.beta1).minimize(self.G_loss, var_list=self.g_vars)

		# d_optim = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate).minimize(self.D_loss, var_list=self.d_vars)
		# g_optim = tf.train.RMSPropOptimizer(learning_rate=config.learning_rate).minimize(self.G_loss, var_list=self.g_vars)
		# clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.d_vars]

		tf.global_variables_initializer().run()
		
		#self.writer = tf.SummaryWriter("./logs", self.sess.graph)
		
		sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
		
		if config.dataset == 'mnist':
			sample_images = data_X[0: self.sample_size]
		
		counter = 1
		start_time = time.time()		
		
		if self.load(self.checkpoint_dir):
			print(" [*] Load SUCESS")
		else:
			print(" [*] Load failed...")
		
		for epoch in xrange(config.epoch):
			if config.dataset == 'mnist':
				batch_idxs = min(len(data_X), config.train_size) // config.batch_size
			else:
				pass

			for idx in xrange(0, batch_idxs):
				if config.dataset == 'mnist':
					batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
				else:
					pass
				
				batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
			
				if config.dataset == 'mnist':
					# Updata D network
					# self.sess.run(clip_D)
					_ = self.sess.run([d_optim],
								feed_dict={self.images:batch_images, self.z:batch_z})
					# _ = self.sess.run([d_optim],
					#  			feed_dict={self.images:batch_images, self.z:batch_z})
	
					# Updata G network twice
					_ = self.sess.run([g_optim],
								feed_dict={self.z:batch_z})
					_ = self.sess.run([g_optim],
								feed_dict={self.z:batch_z})
					# _ = self.sess.run([g_optim],
					# 			feed_dict={self.z:batch_z})
					# _ = self.sess.run([g_optim],
					# 			feed_dict={self.z:batch_z})
					# _ = self.sess.run([g_optim],
					#  			feed_dict={self.z:batch_z})
					
					errD_fake = self.D_loss_fake.eval({self.z:batch_z})
					errD_real = self.D_loss_real.eval({self.images:batch_images})
					errG = self.G_loss.eval({self.z:batch_z})
				else:
					pass
									
				counter += 1
				print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"\
					% (epoch, idx, batch_idxs,
						time.time() - start_time, errD_fake+errD_real, errG))
				if np.mod(counter, 500) == 1:
					samples, d_loss, g_loss = self.sess.run(
						[self.G, self.D_loss, self.G_loss],
						feed_dict={self.z:sample_z, self.images:sample_images})	
					save_images(samples, [8, 8],
						'./{}/train_{:02d}_{:04d}.png'.format(config.sample_dir,
						epoch, idx))
					print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

				if np.mod(counter, 500) == 2:	
					self.save(config.checkpoint_dir, counter)


	def discriminator(self, image, y=None, reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()

			if not y:
				h0 = tf.contrib.layers.conv2d(image, num_outputs=self.df_dim, kernel_size=3,
											stride=2, padding="SAME", activation_fn=lrelu)
				h1 = tf.contrib.layers.conv2d(h0, num_outputs=self.df_dim * 2, kernel_size=3,
											stride=2, padding="SAME", activation_fn=lrelu,
											normalizer_fn=tf.contrib.layers.batch_norm)
				# h2 = tf.contrib.layers.conv2d(h1, num_outputs=self.df_dim * 4, kernel_size=3,
				# 							stride=2, padding="SAME", activation_fn=lrelu,
				# 							normalizer_fn=tf.contrib.layers.batch_norm)
				# h3 = tf.contrib.layers.conv2d(h2, num_outputs=self.df_dim * 8, kernel_size=3,
				# 							stride=2, padding="SAME", activation_fn=lrelu,
				# 							normalizer_fn=tf.contrib.layers.batch_norm)
				h1 = tf.contrib.layers.flatten(h1)
				h2 = tf.contrib.layers.fully_connected(h1, 128, activation_fn=lrelu, normalizer_fn=tf.contrib.layers.batch_norm,
														weights_initializer=tf.random_normal_initializer(0, 0.02))
				h3 = tf.contrib.layers.fully_connected(h2, 1, activation_fn=None,
														weights_initializer=tf.random_normal_initializer(0, 0.02))
				# h4_q1 = tf.contrib.layers.fully_connected(tf.reshape(h3, [self.batch_norm, -1]), 128,
				# 						activation_fn=None, normalizer_fn=tf.contrib.layers.batch_norm)
				# h4_q2 = tf.contrib.layers.fully_connected(h4_q1, 10, activation_fn=tf.)

			return tf.nn.sigmoid(h3), h3

	def generator(self, z, y=None):
		with tf.variable_scope("generator") as scope:
			if not y:
				s = self.output_size
				s2, s4, s8 = int(s/2), int(s/4), int(s/8)
				self.z_ = tf.contrib.layers.fully_connected(z, self.gf_dim*2*s4*s4, activation_fn=tf.nn.relu,
															normalizer_fn=tf.contrib.layers.batch_norm,
															weights_initializer=tf.random_normal_initializer(0, 0.02))
				h0 = tf.reshape(self.z_, [-1, s4, s4, self.gf_dim*2])
				h1 = tf.contrib.layers.conv2d_transpose(h0, self.gf_dim, kernel_size=4, stride=2, padding="SAME",
														activation_fn=tf.nn.relu, normalizer_fn=tf.contrib.layers.batch_norm,
														weights_initializer=tf.random_normal_initializer(0, 0.02))
				h2 = tf.contrib.layers.conv2d_transpose(h1, self.c_dim, kernel_size=4, stride=2, padding="SAME",activation_fn=tf.nn.sigmoid,
														weights_initializer=tf.random_normal_initializer(0, 0.02))
				print (h2)
				return h2
	
	def save(self, checkpoint_dir, step):
		model_name = "DCGAN.model"
		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)


	def load(self, checkpoint_dir):
		print(" [*] Reading checkpoints...")

		model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
		checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			return True
		else:
			return False
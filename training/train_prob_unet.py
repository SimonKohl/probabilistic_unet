# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Probabilistic U-Net training script."""

import tensorflow as tf
import numpy as np
import os
import time
from tqdm import tqdm
import shutil
import logging
import argparse
from importlib.machinery import SourceFileLoader

from data.cityscapes.data_loader import get_train_generators
from model.probabilistic_unet import ProbUNet
import utils.training_utils as training_utils


def train(cf):
	"""Perform training from scratch."""

	# do not use all gpus
	os.environ["CUDA_VISIBLE_DEVICES"] = cf.cuda_visible_devices

	# initialize data providers
	data_provider = get_train_generators(cf)
	train_provider = data_provider['train']
	val_provider = data_provider['val']

	prob_unet = ProbUNet(latent_dim=cf.latent_dim, num_channels=cf.num_channels,
						 num_1x1_convs=cf.num_1x1_convs,
						 num_classes=cf.num_classes, num_convs_per_block=cf.num_convs_per_block,
						 initializers={'w': training_utils.he_normal(),
									   'b': tf.truncated_normal_initializer(stddev=0.001)},
						 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0)})

	x = tf.placeholder(tf.float32, shape=cf.network_input_shape)
	y = tf.placeholder(tf.uint8, shape=cf.label_shape)
	mask = tf.placeholder(tf.uint8, shape=cf.loss_mask_shape)

	global_step = tf.train.get_or_create_global_step()

	if cf.learning_rate_schedule == 'piecewise_constant':
		learning_rate = tf.train.piecewise_constant(x=global_step, **cf.learning_rate_kwargs)
	else:
		learning_rate = tf.train.exponential_decay(learning_rate=cf.initial_learning_rate, global_step=global_step,
											       **cf.learning_rate_kwargs)
	with tf.device(cf.gpu_device):
		prob_unet(x, y, is_training=True, one_hot_labels=cf.one_hot_labels)
		elbo = prob_unet.elbo(y, reconstruct_posterior_mean=cf.use_posterior_mean, beta=cf.beta, loss_mask=mask,
							  analytic_kl=cf.analytic_kl, one_hot_labels=cf.one_hot_labels)
		reconstructed_logits = prob_unet._rec_logits
		sampled_logits = prob_unet.sample()

		reg_loss = cf.regularizarion_weight * tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
		loss = -elbo + reg_loss
		rec_loss = prob_unet._rec_loss_mean
		kl = prob_unet._kl

		mean_val_rec_loss = tf.placeholder(tf.float32, shape=(), name="mean_val_rec_loss")
		mean_val_kl = tf.placeholder(tf.float32, shape=(), name="mean_val_kl")

	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

	# prepare tf summaries
	train_elbo_summary = tf.summary.scalar('train_elbo', elbo)
	train_kl_summary = tf.summary.scalar('train_kl', kl)
	train_rec_loss_summary = tf.summary.scalar('rec_loss', rec_loss)
	train_loss_summary = tf.summary.scalar('train_loss', loss)
	reg_loss_summary = tf.summary.scalar('train_reg_loss', reg_loss)
	lr_summary = tf.summary.scalar('learning_rate', learning_rate)
	beta_summary = tf.summary.scalar('beta', cf.beta)
	training_summary_op = tf.summary.merge([train_loss_summary, reg_loss_summary, lr_summary, train_elbo_summary,
											train_kl_summary, train_rec_loss_summary, beta_summary])
	batches_per_second = tf.placeholder(tf.float32, shape=(), name="batches_per_sec_placeholder")
	timing_summary = tf.summary.scalar('batches_per_sec', batches_per_second)
	val_rec_loss_summary = tf.summary.scalar('val_loss', mean_val_rec_loss)
	val_kl_summary = tf.summary.scalar('val_kl', mean_val_kl)
	validation_summary_op = tf.summary.merge([val_rec_loss_summary, val_kl_summary])

	tf.global_variables_initializer()

	# Add ops to save and restore all the variables.
	saver_hook = tf.train.CheckpointSaverHook(checkpoint_dir=cf.exp_dir, save_steps=cf.save_every_n_steps,
											  saver=tf.train.Saver(save_relative_paths=True))
	# save config
	shutil.copyfile(cf.config_path, os.path.join(cf.exp_dir, 'used_config.py'))

	with tf.train.MonitoredTrainingSession(hooks=[saver_hook]) as sess:
		summary_writer = tf.summary.FileWriter(cf.exp_dir, sess.graph)
		logging.info('Model: {}'.format(cf.exp_dir))
		
		for i in tqdm(range(cf.n_training_batches), disable=cf.disable_progress_bar):

			start_time = time.time()
			train_batch = next(train_provider)
			_, train_summary = sess.run([optimizer, training_summary_op],
										feed_dict={x: train_batch['data'], y: train_batch['seg'],
												   mask: train_batch['loss_mask']})
			summary_writer.add_summary(train_summary, i)
			time_delta = time.time() - start_time
			train_speed = sess.run(timing_summary, feed_dict={batches_per_second: 1. / time_delta})
			summary_writer.add_summary(train_speed, i)

			# validation
			if i % cf.validation['every_n_batches'] == 0:

				train_rec = sess.run(reconstructed_logits, feed_dict={x: train_batch['data'], y: train_batch['seg']})
				image_path = os.path.join(cf.exp_dir,
										  'batch_{}_train_reconstructions.png'.format(i // cf.validation['every_n_batches']))
				training_utils.plot_batch(train_batch, train_rec, num_classes=cf.num_classes,
										  cmap=cf.color_map, out_dir=image_path)

				running_mean_val_rec_loss = 0.
				running_mean_val_kl = 0.

				for j in range(cf.validation['n_batches']):
					val_batch = next(val_provider)
					val_rec, val_sample, val_rec_loss, val_kl =\
						sess.run([reconstructed_logits, sampled_logits, rec_loss, kl],
								  feed_dict={x: val_batch['data'], y: val_batch['seg'], mask: val_batch['loss_mask']})
					running_mean_val_rec_loss += val_rec_loss / cf.validation['n_batches']
					running_mean_val_kl += val_kl / cf.validation['n_batches']

					if j == 0:
						image_path = os.path.join(cf.exp_dir,
												  'batch_{}_val_reconstructions.png'.format(i // cf.validation['every_n_batches']))
						training_utils.plot_batch(val_batch, val_rec,  num_classes=cf.num_classes,
												  cmap=cf.color_map, out_dir=image_path)
						image_path = os.path.join(cf.exp_dir,
												  'batch_{}_val_samples.png'.format(i // cf.validation['every_n_batches']))

						for _ in range(3):
							val_sample_ = sess.run(sampled_logits, feed_dict={x: val_batch['data'], y: val_batch['seg']})
							val_sample = np.concatenate([val_sample, val_sample_], axis=1)

						training_utils.plot_batch(val_batch, val_sample, num_classes=cf.num_classes,
												  cmap=cf.color_map, out_dir=image_path)

				val_summary = sess.run(validation_summary_op, feed_dict={mean_val_rec_loss: running_mean_val_rec_loss,
																		 mean_val_kl: running_mean_val_kl})
				summary_writer.add_summary(val_summary, i)

				if cf.disable_progress_bar:
					logging.info('Evaluating epoch {}/{}: validation loss={}, kl={}'\
								 .format(i, cf.n_training_batches, running_mean_val_rec_loss, running_mean_val_kl))

			sess.run(global_step)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Training of the Probabilistic U-Net')
	parser.add_argument('-c', '--config', type=str, default='prob_unet_config.py',
						help='name of the python script defining the training configuration')
	parser.add_argument('-d', '--data_dir', type=str, default='',
						help="full path to the data, if empty the config's data_dir attribute is used")
	args = parser.parse_args()

	# load config
	cf = SourceFileLoader('cf', args.config).load_module()
	if args.data_dir != '':
		cf.data_dir = args.data_dir

	# prepare experiment directory
	if not os.path.isdir(cf.exp_dir):
		os.mkdir(cf.exp_dir)

	# log to file and console
	log_path = os.path.join(cf.exp_dir, 'train.log')
	logging.basicConfig(filename=log_path, level=logging.INFO)
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.info('Logging to {}'.format(log_path))
	tf.logging.set_verbosity(tf.logging.INFO)

	train(cf)
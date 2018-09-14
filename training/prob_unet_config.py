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
"""CityScapes training config."""

import os
import numpy as np
from collections import OrderedDict
from data.cityscapes.cityscapes_labels import labels as cs_labels_tuple

config_path = os.path.realpath(__file__)

#########################################
#             data-loader    			#
#########################################

data_dir = 'PREPROCESSING_OUTPUT_DIRECTORY_ABSOLUTE_PATH'
resolution = 'quarter'
label_density = 'gtFine'
gt_instances = False
train_cities = ['aachen', 'bochum', 'bremen', 'cologne', 'dusseldorf', 'erfurt', 'hamburg', 'hanover',
				'jena', 'krefeld', 'stuttgart', 'strasbourg', 'tubingen', 'weimar', 'zurich']
val_cities = ['darmstadt', 'monchengladbach', 'ulm', ]

num_classes = 19
batch_size = 10
pre_crop_size = [256, 512]
patch_size = [256, 512]
n_train_batches = None
n_val_batches = 274 // batch_size
n_workers = 5
ignore_label = 255

da_kwargs = {
	'random_crop': True,
	'rand_crop_dist': (patch_size[0] / 2., patch_size[1] / 2.),
	'do_elastic_deform': True,
	'alpha': (0., 800.),
	'sigma': (25., 35.),
	'do_rotation': True,
	'angle_x': (-np.pi / 8., np.pi / 8.),
	'angle_y': (0., 0.),
	'angle_z': (0., 0.),
	'do_scale': True,
	'scale': (0.8, 1.2),
	'border_mode_data': 'constant',
	'border_mode_seg': 'constant',
	'border_cval_seg': ignore_label,
	'gamma_retain_stats': True,
	'gamma_range': (0.7, 1.5),
	'p_gamma': 0.3
}

data_format = 'NCHW'
one_hot_labels = False

#########################################
#            label-switches   			#
#########################################

color_map = {label.trainId:label.color for label in cs_labels_tuple}
color_map[255] = (0.,0.,0.)

trainId2name = {labels.trainId: labels.name for labels in cs_labels_tuple}
name2trainId = {labels.name: labels.trainId for labels in cs_labels_tuple}

label_switches = OrderedDict([('sidewalk', 8./17.), ('person', 7./17.), ('car', 6./17.), ('vegetation', 5./17.), ('road', 4./17.)])
num_classes += len(label_switches)
switched_Id2name = {19+i:list(label_switches.keys())[i] + '_2' for i in range(len(label_switches))}
switched_name2Id = {list(label_switches.keys())[i] + '_2':19+i for i in range(len(label_switches))}
trainId2name = {**trainId2name, **switched_Id2name}
name2trainId = {**name2trainId, **switched_name2Id}

switched_labels2color = {'road_2': (84, 86, 22), 'person_2': (167, 242, 242), 'vegetation_2': (242, 160, 19),
				 		 'car_2': (30, 193, 252), 'sidewalk_2': (46, 247, 180)}
switched_cmap = {switched_name2Id[i]:switched_labels2color[i] for i in switched_name2Id.keys()}
color_map = {**color_map, **switched_cmap}

#########################################
#          network & training			#
#########################################

cuda_visible_devices = '0'
cpu_device = '/cpu:0'
gpu_device = '/gpu:0'

network_input_shape = (None, 3) + tuple(patch_size)
network_output_shape = (None, num_classes) + tuple(patch_size)
label_shape = (None, 1) + tuple(patch_size)
loss_mask_shape = label_shape

base_channels = 32
num_channels = [base_channels, 2*base_channels, 4*base_channels,
				6*base_channels, 6*base_channels, 6*base_channels, 6*base_channels]

num_convs_per_block = 3

n_training_batches = 240000
validation = {'n_batches': n_val_batches, 'every_n_batches': 2000}

learning_rate_schedule = 'piecewise_constant'
learning_rate_kwargs = {'values': [1e-4, 0.5e-4, 1e-5, 0.5e-6],
						'boundaries': [80000, 160000, 240000],
						'name': 'piecewise_constant_lr_decay'}
initial_learning_rate = learning_rate_kwargs['values'][0]

regularizarion_weight = 1e-5
latent_dim = 6
num_1x1_convs = 3
beta = 1.0
analytic_kl = True
use_posterior_mean = False
save_every_n_steps = n_training_batches // 3 if n_training_batches >= 100000 else n_training_batches
disable_progress_bar = False

exp_dir = "EXPERIMENT_OUTPUT_DIRECTORY_ABSOLUTE_PATH"

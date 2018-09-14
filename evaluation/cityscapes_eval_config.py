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
"""Cityscapes evaluation config."""

import os
from collections import OrderedDict
from data.cityscapes.cityscapes_labels import labels as cs_labels_tuple
from model import pretrained_weights

config_path = os.path.realpath(__file__)

#########################################
#                 data      			#
#########################################

data_dir = 'PREPROCESSING_OUTPUT_DIRECTORY_ABSOLUTE_PATH'
resolution = 'quarter'
label_density = 'gtFine'
num_classes = 19
one_hot_labels = False
ignore_label = 255
cities = ['frankfurt', 'lindau', 'munster']

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

exp_modes = len(label_switches)
num_modes = 2 ** exp_modes

#########################################
#               network 			    #
#########################################

cuda_visible_devices = '0'
cpu_device = '/cpu:0'
gpu_device = '/gpu:0'

patch_size = [256, 512]
network_input_shape = (None, 3) +tuple(patch_size)
network_output_shape = (None, num_classes) + tuple(patch_size)
label_shape = (None, 1) + tuple(patch_size)
loss_mask_shape = label_shape

base_channels = 32
num_channels = [base_channels, 2*base_channels, 4*base_channels,
				6*base_channels, 6*base_channels, 6*base_channels, 6*base_channels]

num_convs_per_block = 3

latent_dim = 6
num_1x1_convs = 3
analytic_kl = True
use_posterior_mean = False

#########################################
#             evaluation 			    #
#########################################

num_samples = 16
exp_dir = '/'.join(os.path.abspath(pretrained_weights.__file__).split('/')[:-1])
out_dir = 'EVALUATION_OUTPUT_DIRECTORY_ABSOLUTE_PATH'
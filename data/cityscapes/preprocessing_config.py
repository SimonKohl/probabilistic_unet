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
"""Cityscapes preprocessing config."""

raw_data_dir = 'SET_INPUT_DIRECTORY_ABSOLUTE_PATH_HERE'
out_dir = 'SET_OUTPUT_DIRECTORY_ABSOLUTE_PATH_HERE'

# settings:
settings = {
			'train': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
					  'label_modalities': ['labelIds']},
			'val': {'resolutions': [1.0, 0.5, 0.25], 'label_densities': ['gtFine'],
					'label_modalities': ['labelIds']},
			}

data_format = 'NCHW'

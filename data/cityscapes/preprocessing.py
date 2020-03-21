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
"""Script to preprocess the Cityscapes dataset."""

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import imp

resolution_map = {1.0: 'full', 0.5: 'half', 0.25: 'quarter'}

def resample(img, scale_factor=1.0, interpolation=Image.BILINEAR):
	"""
	Resample PIL.Image objects.
	:param img: PIL.Image object
	:param scale_factor: float
	:param interpolation: PIL.Image interpoaltion method
	:return: PIL.Image object
	"""
	width, height = img.size
	basewidth = width * scale_factor
	basewidth = int(basewidth)
	wpercent = (basewidth / float(width))
	hsize = int((float(height) * wpercent))
	return img.resize((basewidth, hsize), interpolation)

def recursive_mkdir(nested_dir_list):
	"""
	Make the full nested path of directories provided. Order in list implies nesting depth.
	:param nested_dir_list: list of strings
	:return:
	"""
	nested_dir = ''
	for dir in nested_dir_list:
		nested_dir = os.path.join(nested_dir, dir)
		if not os.path.isdir(nested_dir):
			os.mkdir(nested_dir)
	return

def preprocess(cf):

	for set in list(cf.settings.keys()):
		print('Processing {} set.'.format(set))

		# image dir
		image_dir = os.path.join(cf.raw_data_dir, 'leftImg8bit', set)
		city_names = os.listdir(image_dir)

		for city in city_names:
			print('Processing {}'.format(city))
			city_dir = os.path.join(image_dir, city)
			image_names = os.listdir(city_dir)
			image_specifiers = [img.rsplit('_', maxsplit=1)[0] for img in image_names]

			for img_spec in tqdm(image_specifiers):
				for scale in cf.settings[set]['resolutions']:
					recursive_mkdir([cf.out_dir, resolution_map[scale], set, city])

					# image
					img_path = os.path.join(city_dir, img_spec + '_leftImg8bit.png')
					img = Image.open(img_path)
					if scale != 1.0:
						img = resample(img, scale_factor=scale, interpolation=Image.BILINEAR)
					img_out_path = os.path.join(cf.out_dir, resolution_map[scale], set, city, img_spec + '_leftImg8bit.npy')
					img_arr = np.array(img).astype(np.float32)

					channel_axis = 0 if img_arr.shape[0] == 3 else 2
					if cf.data_format == 'NCHW' and channel_axis != 0:
						img_arr = np.transpose(img_arr, axes=[2,0,1])
					np.save(img_out_path, img_arr)

					# labels
					for label_density in cf.settings[set]['label_densities']:
						label_dir = os.path.join(cf.raw_data_dir, label_density, set, city)
						for mod in cf.settings[set]['label_modalities']:
							label_spec = img_spec + '_{}_{}'.format(label_density, mod)
							label_path = os.path.join(label_dir, label_spec + '.png')
							label = Image.open(label_path)
							if scale != 1.0:
								label = resample(label, scale_factor=scale, interpolation=Image.NEAREST)
							label_out_path = os.path.join(cf.out_dir, resolution_map[scale], set, city, label_spec + '.npy')
							np.save(label_out_path, np.array(label).astype(np.uint8))

if __name__ == "__main__":
    cf = imp.load_source('cf', 'preprocessing_config.py')
    preprocess(cf)

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
"""Evaluation metrics tests."""

import pytest
import numpy as np
from evaluation.eval_cityscapes import get_energy_distance_components, get_mode_counts, get_pixelwise_mode_counts
from utils.training_utils import calc_confusion, metrics_from_conf_matrix


def nan_save_array_equal(a, b, nan_replacement=-1.):
	"""Replace NANs to savely compare arrays elementwise."""
	a_nan_ixs = np.where(np.isnan(a))
	a[a_nan_ixs] = nan_replacement

	b_nan_ixs = np.where(np.isnan(b))
	b[b_nan_ixs] = nan_replacement

	return (a == b).all()


@pytest.mark.parametrize("test_input,expected", [
    ([np.zeros(shape=(1,1,10,10)), np.zeros(shape=(1,1,10,10))],
	 {'tp_0': 100, 'tp_1': 0, 'fp_0': 0, 'fp_1': 0, 'fn_0': 0, 'fn_1': 0}),
    ([np.ones(shape=(1,1,10,10)), np.ones(shape=(1,1,10,10))],
	 {'tp_0': 0, 'tp_1': 100, 'fp_0': 0, 'fp_1': 0, 'fn_0': 0, 'fn_1': 0}),
    ([np.ones(shape=(1,1,10,10)), np.zeros(shape=(1,1,10,10))],
	 {'tp_0': 0, 'tp_1': 0, 'fp_0': 100, 'fp_1': 0, 'fn_0': 0, 'fn_1': 100}),
    ([np.concatenate([np.ones(shape=(1,1,10,5)), np.zeros(shape=(1,1,10,5))], axis=-1), np.zeros(shape=(1,1,10,10))],
	 {'tp_0': 50, 'tp_1': 0, 'fp_0': 50, 'fp_1': 0, 'fn_0': 0, 'fn_1': 50}),
])
def test_confusion(test_input, expected):

	gt_seg_modes = test_input[0]
	seg_samples = test_input[1]

	conf_matrix = calc_confusion(gt_seg_modes, seg_samples, class_ixs=[0,1])

	tp_0 = conf_matrix[0,0]
	tp_1 = conf_matrix[1,0]
	fp_0 = conf_matrix[0,1]
	fp_1 = conf_matrix[1,1]
	fn_0 = conf_matrix[0,3]
	fn_1 = conf_matrix[1,3]

	assert tp_0 == expected['tp_0']
	assert tp_1 == expected['tp_1']
	assert fp_0 == expected['fp_0']
	assert fp_1 == expected['fp_1']
	assert fn_0 == expected['fn_0']
	assert fn_1 == expected['fn_1']


@pytest.mark.parametrize("test_input,expected,eval_f", [
    ([np.zeros(shape=(1,1,10,10)), np.zeros(shape=(1,1,10,10))], {'iou_0': 1., 'iou_1': np.nan},
	 [lambda x,y: x == y, lambda x,y: np.isnan(x) and np.isnan(y)]),
	([np.ones(shape=(1,1,10,10)), np.ones(shape=(1,1,10,10))], {'iou_0': np.nan, 'iou_1': 1.},
	 [lambda x,y: np.isnan(x) and np.isnan(y), lambda x,y: x == y]),
	([np.ones(shape=(1,1,10,10)), np.zeros(shape=(1,1,10,10))], {'iou_0': 0., 'iou_1': 0.},
	 [lambda x,y: x == y, lambda x,y: x == y]),
	([np.concatenate([np.ones(shape=(1, 1, 10, 5)), np.zeros(shape=(1, 1, 10, 5))], axis=-1),
	  np.zeros(shape=(1, 1, 10, 10))], {'iou_0': 0.5, 'iou_1': 0.},
	 [lambda x,y: x == y, lambda x,y: x == y])
])
def test_metrics_from_confusion(test_input, expected, eval_f):

	gt_seg_modes = test_input[0]
	seg_samples = test_input[1]

	conf_matrix = calc_confusion(gt_seg_modes, seg_samples, class_ixs=[0,1])
	iou = metrics_from_conf_matrix(conf_matrix)['iou']

	assert eval_f[0](iou[0], expected['iou_0'])
	assert eval_f[1](iou[1], expected['iou_1'])


@pytest.mark.parametrize("test_input,expected,eval_f", [
	([np.zeros(shape=(1,1,1,10,10)), np.zeros(shape=(1,1,1,10,10)), [0]], {'YS': 0., 'SS': 0. , 'YY': 0.},
	 3 * [lambda x,y: x == y]),
	([np.zeros(shape=(1,1,1,10,10)), np.zeros(shape=(1,1,1,10,10)), [1]], {'YS': np.nan, 'SS': np.nan , 'YY': np.nan},
	 3 * [lambda x,y: np.isnan(x) and np.isnan(y)]),
	([np.ones(shape=(1,1,1,10,10)), np.zeros(shape=(1,1,1,10,10)), [0]], {'YS': 1., 'SS': 0. , 'YY': np.nan},
	 2 * [lambda x,y: x == y] + [lambda x,y: np.isnan(x) and np.isnan(y)]),
	([np.concatenate([np.ones(shape=(1,1,1,10,10)), np.zeros(shape=(1,1,1,10,10))], axis=0),
	  np.concatenate([np.zeros(shape=(1,1,1,10,10)), np.ones(shape=(1,1,1,10,10))], axis=0), [0]],
	 {'YS': np.array([[[1.],[np.nan]], [[0.],[1.]]]), 'SS': np.array([[[0.],[1.]], [[1.],[np.nan]]]),
	  'YY': np.array([[[np.nan],[1.]], [[1.],[0.]]])},
	 3 * [lambda x,y: nan_save_array_equal(x,y)]),
	([np.concatenate([np.ones(shape=(1,1,1,10,10)), np.zeros(shape=(1,1,1,10,10))], axis=0),
	  np.concatenate([np.zeros(shape=(1,1,1,10,10)), np.ones(shape=(1,1,1,10,10))], axis=0), 2],
	 {'YS': np.array([[[1.,1.], [np.nan,0.]], [[0.,np.nan], [1.,1.]]]),
	  'SS': np.array([[[0.,np.nan], [1.,1.]], [[1.,1.], [np.nan,0.]]]),
	  'YY': np.array([[[np.nan,0.], [1.,1.]], [[1.,1.], [0.,np.nan]]])},
	 3 * [lambda x,y: nan_save_array_equal(x, y)]),
])
def test_energy_distance_components(test_input, expected, eval_f):

	gt_seg_modes = test_input[0]
	seg_samples = test_input[1]
	eval_class_ids = test_input[2]

	results = get_energy_distance_components(gt_seg_modes, seg_samples, eval_class_ids=eval_class_ids)

	assert eval_f[0](results['YS'], expected['YS'])
	assert eval_f[1](results['SS'], expected['SS'])
	assert eval_f[2](results['YY'], expected['YY'])


@pytest.mark.parametrize("test_input,expected,eval_f", [
	(np.concatenate([np.zeros(shape=(1,5,5)), 0.1 * np.ones(shape=(1,5,5))]), [5,0], lambda x,y: (x==y).all())
])
def test_get_mode_counts(test_input, expected, eval_f):
	mode_counts = get_mode_counts(test_input)

	assert eval_f(mode_counts, expected)


@pytest.mark.parametrize("test_input,expected,eval_f", [
	([np.zeros(shape=(1,1,10,10)),
	  np.concatenate([np.ones(shape=(1,1,1,10,5)), np.zeros(shape=(1,1,1,10,5))], axis=-1), 1],
	 [[100, 50, 50]], lambda x,y: (x==y).all()),
	([np.concatenate([np.ones(shape=(1,1,10,5)), np.zeros(shape=(1,1,10,5))], axis=-1),
	  np.concatenate([np.ones(shape=(1,1,1,10,10)), np.zeros(shape=(1,1,1,10,10))], axis=0), 2],
	 [[100, 50, 0],[100, 50, 0]], lambda x,y: (x==y).all()),
	([np.concatenate([np.ones(shape=(1, 1, 10, 5)), np.zeros(shape=(1, 1, 10, 5))], axis=-1),
	  np.concatenate([3 * np.ones(shape=(1, 1, 1, 10, 10)), 2 * np.ones(shape=(1, 1, 1, 10, 10)),
					  np.ones(shape=(1, 1, 1, 10, 10)), np.zeros(shape=(1, 1, 1, 10, 10))], axis=0), 2],
	 [[200, 50, 50], [200, 50, 50]], lambda x, y: (x == y).all()),
])
def test_pixelwise_mode_counts(test_input, expected, eval_f):

	class cf():
		def __init__(self, num_classes):
			self.label_switches = {'class_{}'.format(i): np.random.uniform(0.0,1.0) for i in range(num_classes)}
			self.name2trainId = {**{'class_{}'.format(i): i for i in range(num_classes)},
				         		 **{'class_{}_2'.format(i): i + num_classes for i in range(num_classes)}}
	num_classes = test_input[2]
	pixelwise_mode_counts = get_pixelwise_mode_counts(cf(num_classes), seg=test_input[0], seg_samples=test_input[1])

	assert eval_f(pixelwise_mode_counts, expected)
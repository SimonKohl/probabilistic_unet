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
"""Training utilities."""

import tensorflow as tf
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.python.ops.init_ops import VarianceScaling


def ce_loss(labels, logits, n_classes, loss_mask=None, data_format='NCHW', one_hot_labels=True, name='ce_loss'):
    """
    Cross-entropy loss.
    :param labels: 4D tensor
    :param logits: 4D tensor
    :param n_classes: integer for number of classes
    :param loss_mask: binary 4D tensor, pixels to mask should be marked by 1s
    :param data_format: string
    :param one_hot_labels: bool, indicator for whether labels are to be expected in one-hot representation
    :param name: string
    :return: dict of (pixel-wise) mean and sum of cross-entropy loss
    """
    with tf.variable_scope(name):
        # permute class channels into last axis
        if data_format == 'NCHW':
            labels = tf.transpose(labels, [0,2,3,1])
            logits = tf.transpose(logits, [0,2,3,1])
        elif data_format == 'NCDHW':
            labels = tf.transpose(labels, [0,2,3,4,1])
            logits = tf.transpose(logits, [0,2,3,4,1])

        batch_size = tf.cast(tf.shape(labels)[0], tf.float32)

        if one_hot_labels:
            flat_labels = tf.reshape(labels, [-1, n_classes])
        else:
            flat_labels = tf.reshape(labels, [-1])
            flat_labels = tf.one_hot(indices=flat_labels, depth=n_classes, axis=-1)
        flat_logits = tf.reshape(logits, [-1, n_classes])

        # do not compute gradients wrt the labels
        flat_labels = tf.stop_gradient(flat_labels)

        ce_per_pixel = tf.nn.softmax_cross_entropy_with_logits_v2(labels=flat_labels, logits=flat_logits)

        # optional element-wise masking with binary loss mask
        if loss_mask is None:
            ce_sum = tf.reduce_sum(ce_per_pixel) / batch_size
            ce_mean = tf.reduce_mean(ce_per_pixel)
        else:
            loss_mask_flat = tf.reshape(loss_mask, [-1,])
            loss_mask_flat = (1. - tf.cast(loss_mask_flat, tf.float32))
            ce_sum = tf.reduce_sum(loss_mask_flat * ce_per_pixel) / batch_size
            n_valid_pixels = tf.reduce_sum(loss_mask_flat)
            ce_mean = tf.reduce_sum(loss_mask_flat * ce_per_pixel) / n_valid_pixels

        return {'sum': ce_sum, 'mean': ce_mean}


def softmax_2_onehot(arr):
    """Transform a numpy array of softmax values into a one-hot encoded array. Assumes classes are encoded in axis 1.
    :param arr: ND array
    :return: ND array
    """
    num_classes = arr.shape[1]
    arr_argmax = np.argmax(arr, axis=1)

    for c in range(num_classes):
        arr[:,c] = (arr_argmax == c).astype(np.uint8)
    return arr


def numpy_one_hot(label_arr, num_classes):
    """One-hotify an integer-labeled numpy array. One-hot encoding is encoded in additional last axis.
    :param label_arr: ND array
    :param num_classes: integer
    :return: (N+1)D array
    """
    # replace labels >= num_classes with 0
    label_arr[label_arr >= num_classes] = 0

    res = np.eye(num_classes)[np.array(label_arr).reshape(-1)]
    return res.reshape(list(label_arr.shape)+[num_classes])


def calc_confusion(labels, samples, class_ixs, loss_mask=None):
    """
    Compute confusion matrix for each class across the given arrays.
    Assumes classes are given in integer-valued encoding.
    :param labels: 4/5D array
    :param samples: 4/5D array
    :param class_ixs: integer or list of integers specifying the classes to evaluate
    :param loss_mask: 4/5D array
    :return: 2D array
    """
    try:
        assert labels.shape == samples.shape
    except:
        raise AssertionError('shape mismatch {} vs. {}'.format(labels.shape, samples.shape))

    if isinstance(class_ixs, int):
        num_classes = class_ixs
        class_ixs = range(class_ixs)
    elif isinstance(class_ixs, list):
        num_classes = len(class_ixs)
    else:
        raise TypeError('arg class_ixs needs to be int or list, not {}.'.format(type(class_ixs)))

    if loss_mask is None:
        shp = labels.shape
        loss_mask = np.zeros(shape=(shp[0], 1, shp[2], shp[3]))

    conf_matrix = np.zeros(shape=(num_classes, 4), dtype=np.float32)
    for i,c in enumerate(class_ixs):

        pred_ = (samples == c).astype(np.uint8)
        labels_ = (labels == c).astype(np.uint8)

        conf_matrix[i,0] = int(((pred_ != 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # TP
        conf_matrix[i,1] = int(((pred_ != 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # FP
        conf_matrix[i,2] = int(((pred_ == 0) * (labels_ == 0) * (loss_mask != 1)).sum()) # TN
        conf_matrix[i,3] = int(((pred_ == 0) * (labels_ != 0) * (loss_mask != 1)).sum()) # FN

    return conf_matrix


def metrics_from_conf_matrix(conf_matrix):
    """
    Calculate IoU per class from a confusion_matrix.
    :param conf_matrix: 2D array of shape (num_classes, 4)
    :return: dict holding 1D-vectors of metrics
    """
    tps = conf_matrix[:,0]
    fps = conf_matrix[:,1]
    fns = conf_matrix[:,3]

    metrics = {}
    metrics['iou'] = np.zeros_like(tps, dtype=np.float32)

    # iterate classes
    for c in range(tps.shape[0]):
        # unless both the prediction and the ground-truth is empty, calculate a finite IoU
        if tps[c] + fps[c] + fns[c] != 0:
            metrics['iou'][c] = tps[c] / (tps[c] + fps[c] + fns[c])
        else:
            metrics['iou'][c] = np.nan

    return metrics


def he_normal(seed=None):
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    Arguments:
        seed: A Python integer. Used to seed the random generator.
    Returns:
          An initializer.
    References:
          He et al., http://arxiv.org/abs/1502.01852
    Code:
        https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/keras/initializers.py
    """
    return VarianceScaling(scale=2., mode='fan_in', distribution='normal', seed=seed)


def plot_batch(batch, prediction, cmap, num_classes, out_dir=None, clip_range=True):
    """
    Plots a batch of images, segmentations & samples and optionally saves it to disk.
    :param batch: dict holding images and gt labels for a batch
    :param prediction: logit prediction of the corresponding batch
    :param cmap: dictionary as colormap
    :param out_dir: full path to save png image to
    :return:
    """
    img_arr = batch['data']
    seg_arr = batch['seg']

    num_predictions = prediction.shape[1] // num_classes
    num_y_tiles = 2 + num_predictions
    batch_size = img_arr.shape[0]

    f = plt.figure(figsize=(batch_size * 4, num_y_tiles * 2))
    gs = gridspec.GridSpec(num_y_tiles, batch_size, wspace=0.0, hspace=0.0)

    # suppress matplotlib range warnings
    if clip_range:
        img_arr[img_arr < 0.] = 0.
        img_arr[img_arr > 1.] = 1.

    for tile in range(batch_size):
        # image
        ax = plt.subplot(gs[0, tile])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(np.transpose(img_arr[tile], axes=[1,2,0]))

        # (here sampled) gt segmentation
        ax = plt.subplot(gs[1, tile])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if seg_arr.shape[1] == 1:
            gt_seg = np.squeeze(seg_arr[tile], axis=0)
        else:
            gt_seg = np.argmax(seg_arr[tile], axis=0)
        plt.imshow(to_rgb(gt_seg, cmap))

        # multiple predictions can be concatenated in channel axis, iterate all predictions
        for i in range(num_predictions):
            ax = plt.subplot(gs[2 + i, tile])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            single_prediction = prediction[tile][i * num_classes: (i+1) * num_classes]
            pred_seg = np.argmax(single_prediction, axis=0)
            plt.imshow(to_rgb(pred_seg, cmap))
    if out_dir is not None:
        plt.savefig(out_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
    plt.close(f)


def to_rgb(arr, cmap):
    """
    Transform an integer-labeled segmentation map using an rgb color-map.
    :param arr: img_arr w/o a color-channel
    :param cmap: dictionary mapping from integer class labels to rgb values
    :return:
    """
    new_arr = np.zeros(shape=(arr.shape)+(3,))
    for c in cmap.keys():
        ixs = np.where(arr == c)
        new_arr[ixs] = [cmap[c][i] / 255. for i in range(3)]
    return new_arr

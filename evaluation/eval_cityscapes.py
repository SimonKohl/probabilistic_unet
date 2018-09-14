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
"""Cityscapes evaluation script."""

import tensorflow as tf
import numpy as np
import os
import argparse
from tqdm import tqdm
from multiprocessing import Process, Queue
from importlib.machinery import SourceFileLoader
import logging
import pickle

from model.probabilistic_unet import ProbUNet
from data.cityscapes.data_loader import loadFiles, map_labels_to_trainId
from utils import training_utils


def write_test_predictions(cf):
    """
    Write samples as numpy arrays.
    :param cf: config module
    :return:
    """
    # do not use all gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = cf.cuda_visible_devices

    data_dir = os.path.join(cf.data_dir, cf.resolution)
    data_dict = loadFiles(label_density=cf.label_density, split='val', input_path=data_dir,
                          cities=None, instance=False)
    # prepare out_dir
    if not os.path.isdir(cf.out_dir):
        os.mkdir(cf.out_dir)

    logging.info('Writing to {}'.format(cf.out_dir))

    # initialize computation graph
    prob_unet = ProbUNet(latent_dim=cf.latent_dim, num_channels=cf.num_channels,
                         num_1x1_convs=cf.num_1x1_convs,
                         num_classes=cf.num_classes, num_convs_per_block=cf.num_convs_per_block,
                         initializers={'w': training_utils.he_normal(),
                                       'b': tf.truncated_normal_initializer(stddev=0.001)},
                         regularizers={'w': tf.contrib.layers.l2_regularizer(1.0),
                                       'b': tf.contrib.layers.l2_regularizer(1.0)})
    x = tf.placeholder(tf.float32, shape=cf.network_input_shape)

    with tf.device(cf.gpu_device):
        prob_unet(x, is_training=False, one_hot_labels=cf.one_hot_labels)
        sampled_logits = prob_unet.sample()

    saver = tf.train.Saver(save_relative_paths=True)
    with tf.train.MonitoredTrainingSession() as sess:

        print('EXP DIR', cf.exp_dir)
        latest_ckpt_path = tf.train.latest_checkpoint(cf.exp_dir)
        print('CKPT PATH', latest_ckpt_path)
        saver.restore(sess, latest_ckpt_path)

        for k, v in tqdm(data_dict.items()):
            img = np.load(v['data']) / 255.
            # add batch dimensions
            img = img[np.newaxis]

            for i in range(cf.num_samples):
                sample = sess.run(sampled_logits, feed_dict={x: img})
                sample = np.argmax(sample, axis=1)[:, np.newaxis]
                sample = sample.astype(np.uint8)
                sample_path = os.path.join(cf.out_dir, '{}_sample{}_labelIds.npy'.format(k, i))
                np.save(sample_path, sample)


def get_array_of_modes(cf, seg):
    """
    Assemble an array holding all label modes.
    :param cf: config module
    :param seg: 4D integer array
    :return: 4D integer array
    """
    mode_stats = get_mode_statistics(cf.label_switches, exp_modes=cf.exp_modes)
    switch = mode_stats['switch']

    # construct ground-truth modes
    gt_seg_modes = np.zeros(shape=(cf.num_modes,) + seg.shape, dtype=np.uint8)
    for mode in range(cf.num_modes):
        switched_seg = seg.copy()
        for i, c in enumerate(cf.label_switches.keys()):
            if switch[mode, i]:
                init_id = cf.name2trainId[c]
                final_id = cf.name2trainId[c + '_2']
                switched_seg[switched_seg == init_id] = final_id
        gt_seg_modes[mode] = switched_seg

    return gt_seg_modes


def get_array_of_samples(cf, img_key):
    """
    Assemble an array holding all segmentation samples for a given image.
    :param cf: config module
    :param img_key: string
    :return: 5D integer array
    """
    seg_samples = np.zeros(shape=(cf.num_samples,1,1) + tuple(cf.patch_size), dtype=np.uint8)
    for i in range(cf.num_samples):
        sample_path = os.path.join(cf.out_dir, '{}_sample{}_labelIds.npy'.format(img_key, i))
        try:
            seg_samples[i] = np.load(sample_path)
        except:
            print('Could not load {}'.format(sample_path))

    return seg_samples


def get_mode_counts(d_matrix_YS):
    """
    Calculate image-level mode counts.
    :param d_matrix_YS: 3D array
    :return: numpy array
    """
    # assign each sample to a mode
    mean_d = np.nanmean(d_matrix_YS, axis=-1)
    sampled_modes = np.argmin(mean_d, axis=-2)

    # count the modes
    num_modes = d_matrix_YS.shape[0]
    mode_count = np.zeros(shape=(num_modes,), dtype=np.int)
    for sampled_mode in sampled_modes:
        mode_count[sampled_mode] += 1

    return mode_count


def get_pixelwise_mode_counts(cf, seg, seg_samples):
    """
    Calculate pixel-wise mode counts.
    :param cf: config module
    :param seg: 4D array of integer labeled segmentations
    :param seg_samples: 5D array of integer labeled segmentations
    :return: array of shape (switchable classes, 3)
    """
    assert seg.shape == seg_samples.shape[1:]
    num_samples = seg_samples.shape[0]
    pixel_counts = np.zeros(shape=(len(cf.label_switches),3), dtype=np.int)

    # iterate all switchable classes
    for i,c in enumerate(cf.label_switches.keys()):
        c_id = cf.name2trainId[c]
        alt_c_id = cf.name2trainId[c+'_2']
        c_ixs = np.where(seg == c_id)

        total_num_pixels = np.sum((seg == c_id).astype(np.uint8)) * num_samples
        pixel_counts[i,0] = total_num_pixels

        # count the pixels of original class|original class and alternative class|original class
        for j in range(num_samples):
            sample = seg_samples[j]
            sampled_original_pixels = np.sum((sample[c_ixs] == c_id).astype(np.uint8))
            sampled_alternative_pixels = np.sum((sample[c_ixs] == alt_c_id).astype(np.uint8))
            pixel_counts[i,1] += sampled_original_pixels
            pixel_counts[i,2] += sampled_alternative_pixels

    return pixel_counts


def get_mode_statistics(label_switches, exp_modes=5):
    """
    Calculate a binary matrix of switches as well as a vector of mode probabilities.
    :param label_switches: dict specifying class names and their individual sampling probabilities
    :param exp_modes: integer, number of independently switchable classes
    :return: dict
    """
    num_modes = 2 ** exp_modes

    # assemble a binary matrix of switch decisions
    switch = np.zeros(shape=(num_modes, 5), dtype=np.uint8)
    for i in range(exp_modes):
        switch[:,i] = 2 ** i * (2 ** (exp_modes - 1 - i) * [0] + 2 ** (exp_modes - 1 - i) * [1])

    # calculate the probability for each individual mode
    mode_probs = np.zeros(shape=(num_modes,), dtype=np.float32)
    for mode in range(num_modes):
        prob = 1.
        for i, c in enumerate(label_switches.keys()):
            if switch[mode, i]:
                prob *= label_switches[c]
            else:
                prob *= 1. - label_switches[c]
        mode_probs[mode] = prob
    assert np.sum(mode_probs) == 1.

    return {'switch': switch, 'mode_probs': mode_probs}


def get_energy_distance_components(gt_seg_modes, seg_samples, eval_class_ids, ignore_mask=None):
    """
    Calculates the components for the IoU-based generalized energy distance given an array holding all segmentation
    modes and an array holding all sampled segmentations.
    :param gt_seg_modes: N-D array in format (num_modes,[...],H,W)
    :param seg_samples: N-D array in format (num_samples,[...],H,W)
    :param eval_class_ids: integer or list of integers specifying the classes to encode, if integer range() is applied
    :param ignore_mask: N-D array in format ([...],H,W)
    :return: dict
    """
    num_modes = gt_seg_modes.shape[0]
    num_samples = seg_samples.shape[0]

    if isinstance(eval_class_ids, int):
        eval_class_ids = list(range(eval_class_ids))

    d_matrix_YS = np.zeros(shape=(num_modes, num_samples, len(eval_class_ids)), dtype=np.float32)
    d_matrix_YY = np.zeros(shape=(num_modes, num_modes, len(eval_class_ids)), dtype=np.float32)
    d_matrix_SS = np.zeros(shape=(num_samples, num_samples, len(eval_class_ids)), dtype=np.float32)

    # iterate all ground-truth modes
    for mode in range(num_modes):

        ##########################################
        #   Calculate d(Y,S) = [1 - IoU(Y,S)],	 #
        #   with S ~ P_pred, Y ~ P_gt  			 #
        ##########################################

        # iterate the samples S
        for i in range(num_samples):
            conf_matrix = training_utils.calc_confusion(gt_seg_modes[mode], seg_samples[i],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = training_utils.metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YS[mode, i] = 1. - iou

        ###########################################
        #   Calculate d(Y,Y') = [1 - IoU(Y,Y')],  #
        #   with Y,Y' ~ P_gt  	   				  #
        ###########################################

        # iterate the ground-truth modes Y' while exploiting the pair-wise symmetries for efficiency
        for mode_2 in range(mode, num_modes):
            conf_matrix = training_utils.calc_confusion(gt_seg_modes[mode], gt_seg_modes[mode_2],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = training_utils.metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_YY[mode, mode_2] = 1. - iou
            d_matrix_YY[mode_2, mode] = 1. - iou

    #########################################
    #   Calculate d(S,S') = 1 - IoU(S,S'),  #
    #   with S,S' ~ P_pred        			#
    #########################################

    # iterate all samples S
    for i in range(num_samples):
        # iterate all samples S'
        for j in range(i, num_samples):
            conf_matrix = training_utils.calc_confusion(seg_samples[i], seg_samples[j],
                                                        loss_mask=ignore_mask, class_ixs=eval_class_ids)
            iou = training_utils.metrics_from_conf_matrix(conf_matrix)['iou']
            d_matrix_SS[i, j] = 1. - iou
            d_matrix_SS[j, i] = 1. - iou

    return {'YS': d_matrix_YS, 'SS': d_matrix_SS, 'YY': d_matrix_YY}


def calc_energy_distances(d_matrices, num_samples=None, probability_weighted=False, label_switches=None, exp_mode=5):
    """
    Calculate the energy distance for each image based on matrices holding the combinatorial distances.
    :param d_matrices: dict holding 4D arrays of shape \
    (num_images, num_modes/num_samples, num_modes/num_samples, num_classes)
    :param num_samples: integer or None
    :param probability_weighted: bool
    :param label_switches: None or dict
    :param exp_mode: integer
    :return: numpy array
    """
    d_matrices = d_matrices.copy()

    if num_samples is None:
        num_samples = d_matrices['SS'].shape[1]

    d_matrices['YS'] = d_matrices['YS'][:,:,:num_samples]
    d_matrices['SS'] = d_matrices['SS'][:,:num_samples,:num_samples]

    # perform a nanmean over the class axis so as to not factor in classes that are not present in
    # both the ground-truth mode as well as the sampled prediction
    if probability_weighted:
       mode_stats = get_mode_statistics(label_switches, exp_modes=exp_mode)
       mode_probs = mode_stats['mode_probs']

       mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
       mean_d_YS = np.mean(mean_d_YS, axis=2)
       mean_d_YS = mean_d_YS * mode_probs[np.newaxis, :]
       d_YS = np.sum(mean_d_YS, axis=1)

       mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
       d_SS = np.mean(mean_d_SS, axis=(1, 2))

       mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
       mean_d_YY = mean_d_YY * mode_probs[np.newaxis, :, np.newaxis] * mode_probs[np.newaxis, np.newaxis, :]
       d_YY = np.sum(mean_d_YY, axis=(1, 2))

    else:
       mean_d_YS = np.nanmean(d_matrices['YS'], axis=-1)
       d_YS = np.mean(mean_d_YS, axis=(1,2))

       mean_d_SS = np.nanmean(d_matrices['SS'], axis=-1)
       d_SS = np.mean(mean_d_SS, axis=(1, 2))

       mean_d_YY = np.nanmean(d_matrices['YY'], axis=-1)
       d_YY = np.nanmean(mean_d_YY, axis=(1, 2))

    return 2 * d_YS - d_SS - d_YY


def eval(cf, cities, queue=None, ixs=None):
    """
    Perform evaluation w.r.t the generalized energy distance based on the IoU as well as image-level and pixel-level
    mode frequencies (using samples written to file).
    :param cf: config module
    :param cities: string or list of strings
    :param queue: instance of multiprocessing.Queue
    :param ixs: None or 2-tuple of ints
    :return: NoneType or numpy array
    """
    data_dir = os.path.join(cf.data_dir, cf.resolution)
    data_dict = loadFiles(label_density=cf.label_density, split='val', input_path=data_dir, cities=cities,
                          instance=False)

    num_modes = cf.num_modes
    num_samples = cf.num_samples

    # evaluate only switchable classes, so a total of 10 here
    eval_class_names = list(cf.label_switches.keys()) + list(cf.switched_name2Id.keys())
    eval_class_ids = [cf.name2trainId[n] for n in eval_class_names]
    d_matrices = {'YS': np.zeros(shape=(len(data_dict), num_modes, num_samples, len(eval_class_ids)),
                                 dtype=np.float32),
                  'YY': np.ones(shape=(len(data_dict), num_modes, num_modes, len(eval_class_ids)),
                                dtype=np.float32),
                  'SS': np.ones(shape=(len(data_dict), num_samples, num_samples, len(eval_class_ids)),
                                dtype=np.float32)}
    sampled_mode_counts = np.zeros(shape=(num_modes,), dtype=np.int)
    sampled_pixel_counts = np.zeros(shape=(len(cf.label_switches), 3), dtype=np.int)

    logging.info('Evaluating class names: {} (corresponding to labels {})'.format(eval_class_names, eval_class_ids))

    # allow for data selection by indexing via ixs
    if ixs is None:
        data_keys = list(data_dict.keys())
    else:
        data_keys = list(data_dict.keys())[ixs[0]:ixs[1]]
        for k in d_matrices.keys():
            d_matrices[k] = d_matrices[k][:ixs[1]-ixs[0]]

    # iterate all validation images
    for img_n, img_key in enumerate(tqdm(data_keys)):

        seg = np.load(data_dict[img_key]['seg'])
        seg = map_labels_to_trainId(seg)
        seg = seg[np.newaxis, np.newaxis]
        ignore_mask = (seg == cf.ignore_label).astype(np.uint8)

        seg_samples = get_array_of_samples(cf, img_key)
        gt_seg_modes = get_array_of_modes(cf, seg)

        energy_dist = get_energy_distance_components(gt_seg_modes=gt_seg_modes, seg_samples=seg_samples,
                                                     eval_class_ids=eval_class_ids, ignore_mask=ignore_mask)
        sampled_mode_counts += get_mode_counts(energy_dist['YS'])
        sampled_pixel_counts += get_pixelwise_mode_counts(cf, seg, seg_samples)

        for k in d_matrices.keys():
            d_matrices[k][img_n] = energy_dist[k]

    results = {'d_matrices': d_matrices, 'sampled_pixel_counts': sampled_pixel_counts,
               'sampled_mode_counts': sampled_mode_counts, 'total_num_samples': len(data_keys) * num_samples}

    if queue is not None:
        queue.put(results)
        return
    else:
        return results


def runInParallel(fns_args, queue):
    """Run functions in parallel.
    :param fns_args: list of tuples containing functions and a tuple of arguments each
    :param queue: instance of multiprocessing.Queue()
    :return: list of queue results
    """
    proc = []
    for fn in fns_args:
        p = Process(target=fn[0], args=fn[1])
        p.start()
        proc.append(p)
    return [queue.get() for p in proc]


def multiprocess_evaluation(cf):
    """Evaluate the energy distance in multiprocessing.
    :param cf: config module"""
    q = Queue()
    results = runInParallel([(eval, (cf, 'lindau', q)),
                             (eval, (cf, 'frankfurt', q, (0, 100))),
                             (eval, (cf, 'frankfurt', q, (100, 200))),
                             (eval, (cf, 'frankfurt', q, (200, 267))),
                             (eval, (cf, 'munster', q, (0, 100))),
                             (eval, (cf, 'munster', q, (100, 174)))],
                             queue=q)
    total_num_samples = 0
    sampled_mode_counts = np.zeros(shape=(cf.num_modes,), dtype=np.int)
    sampled_pixel_counts = np.zeros(shape=(len(cf.label_switches), 3), dtype=np.int)
    d_matrices = {'YS':[], 'SS':[], 'YY':[]}

    # aggregate results from the queue
    for result_dict in results:
        for key in d_matrices.keys():
            d_matrices[key].append(result_dict['d_matrices'][key])

        sampled_pixel_counts += result_dict['sampled_pixel_counts']
        sampled_mode_counts += result_dict['sampled_mode_counts']
        total_num_samples += result_dict['total_num_samples']

    for key in d_matrices.keys():
        d_matrices[key] = np.concatenate(d_matrices[key], axis=0)

    # calculate frequencies
    print('pixel frequencies', sampled_pixel_counts)
    sampled_pixelwise_mode_per_class = sampled_pixel_counts[:,1:]
    total_num_pixels_per_class = sampled_pixel_counts[:,0:1]
    sampled_pixel_frequencies = sampled_pixelwise_mode_per_class / total_num_pixels_per_class
    sampled_mode_frequencies = sampled_mode_counts / total_num_samples

    print('sampled pixel frequencies', sampled_pixel_frequencies)
    print('sampled_mode_frequencies', sampled_mode_frequencies)

    results_dict = {'d_matrices': d_matrices, 'pixel_frequencies': sampled_pixel_frequencies,
               'mode_frequencies': sampled_mode_frequencies}

    results_file = os.path.join(cf.out_dir, 'eval_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f, pickle.HIGHEST_PROTOCOL)
    logging.info('Wrote to {}'.format(results_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation step selection.')
    parser.add_argument('--write_samples', dest='write_samples', action='store_true')
    parser.add_argument('--eval_samples', dest='write_samples', action='store_false')
    parser.set_defaults(write_samples=True)
    parser.add_argument('-c', '--config_name', type=str, default='cityscapes_eval_config.py',
                        help='name of the python file that is loaded as config module')
    args = parser.parse_args()

    # load evaluation config
    cf = SourceFileLoader('cf', args.config_name).load_module()

    # prepare evaluation directory
    if not os.path.isdir(cf.out_dir):
        os.mkdir(cf.out_dir)

    # log to file and console
    log_path = os.path.join(cf.out_dir, 'eval.log')
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info('Logging to {}'.format(log_path))

    if args.write_samples:
        logging.info('Writing samples to {}'.format(cf.out_dir))
        write_test_predictions(cf)
    else:
        logging.info('Evaluating samples from {}'.format(cf.out_dir))
        multiprocess_evaluation(cf)
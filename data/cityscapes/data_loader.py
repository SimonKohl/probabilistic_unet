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
"""Script to serve the CityScapes dataset."""

import os
import sys, glob
import numpy as np
import imp
import logging

from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
from batchgenerators.transforms import AbstractTransform
from .cityscapes_labels import labels as cityscapes_labels_tuple

def loadFiles(label_density, split, input_path, cities=None, instance=False):
    """
    Assemble dict of file paths.
    :param label_density: string in ['gtFine', 'gtCoarse']
    :param split: string in ['train', 'val', 'test', 'train_extra']
    :param input_path: string
    :param cities: list of strings or None
    :param instance: bool
    :return: dict
    """
    input_dir = os.path.join(input_path, split)
    logging.info("Assembling file dict from {}.".format(input_dir))
    paths_dict = {}
    for path, dirs, files in os.walk(input_dir):

        skip_city = False
        if cities is not None:
            current_city = path.rsplit('/', 1)[-1]
            if current_city not in cities:
                skip_city = True

        if not skip_city:
            logging.info('Reading from {}'.format(path))
            label_paths, img_paths = searchFiles(path, label_density, instance)
            paths_dict = {**paths_dict, **zipPaths(label_paths, img_paths)}

    return paths_dict


def searchFiles(path, label_density, instance=False):
    """
    Get file paths via wildcard search.
    :param path: path to files for each city
    :param label_density: string in ['gtFine', 'gtCoarse']
    :param instance: bool
    :return: 2 lists
    """
    if (instance == True):
        label_wildcard_search = os.path.join(path, "*{}_instanceIDs.npy".format(label_density))
    else:
        label_wildcard_search = os.path.join(path, "*{}_labelIds.npy".format(label_density))
    label_paths = glob.glob(label_wildcard_search)
    label_paths.sort()
    img_wildcard_search = os.path.join(path, "*_leftImg8bit.npy")
    img_paths = glob.glob(img_wildcard_search)
    img_paths.sort()
    return label_paths, img_paths


def zipPaths(label_paths, img_paths):
    """
    zip paths in form of dict.
    :param label_paths: list of strings
    :param img_paths: list of strings
    :return: dict
    """
    try:
        assert len(label_paths) == len(img_paths)
    except:
        raise Exception('Missmatch: {} label paths vs. {} img paths!'.format(len(label_paths), len(img_paths)))

    paths_dict = {}
    for i, img_path in enumerate(img_paths):
        img_spec = ('_').join(img_paths[i].split('/')[-1].split('_')[:-1])
        try:
            assert img_spec in label_paths[i]
        except:
            raise Exception('img and label name mismatch: {} vs. {}'.format(img_paths[i], label_paths[i]))

        paths_dict[img_spec] = {"data": img_paths[i], "seg": label_paths[i], 'img_spec': img_spec}
    return paths_dict


def augment_gamma(data, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False, retain_stats=False, p_per_sample=0.3):
    """code by Fabian Isensee, see MIC_DKFZ/batch_generators on github."""
    for sample in range(data.shape[0]):
        if np.random.uniform() < p_per_sample:
            if invert_image:
                data = - data
            if not per_channel:
                if retain_stats:
                    mn = data[sample].mean()
                    sd = data[sample].std()
                if np.random.random() < 0.5 and gamma_range[0] < 1:
                    gamma = np.random.uniform(gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                minm = data[sample].min()
                rnge = data[sample].max() - minm
                data[sample] = np.power(((data[sample] - minm) / float(rnge + epsilon)), gamma) * rnge + minm
                if retain_stats:
                    data[sample] = data[sample] - data[sample].mean() + mn
                    data[sample] = data[sample] / (data[sample].std() + 1e-8) * sd
            else:
                for c in range(data.shape[1]):
                    if retain_stats:
                        mn = data[sample][c].mean()
                        sd = data[sample][c].std()
                    if np.random.random() < 0.5 and gamma_range[0] < 1:
                        gamma = np.random.uniform(gamma_range[0], 1)
                    else:
                        gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
                    minm = data[sample][c].min()
                    rnge = data[sample][c].max() - minm
                    data[sample][c] = np.power(((data[sample][c] - minm) / float(rnge + epsilon)), gamma) * rnge + minm
                    if retain_stats:
                        data[sample][c] = data[sample][c] - data[sample][c].mean() + mn
                        data[sample][c] = data[sample][c] / (data[sample][c].std() + 1e-8) * sd
    return data


class GammaTransform(AbstractTransform):
    """Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

    Args:
        gamma_range (tuple of float): range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified)

        invert_image: whether to invert the image before applying gamma augmentation

        retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation

    """

    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=0.3, mask_channel_in_seg=None):
        self.mask_channel_in_seg = mask_channel_in_seg
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        data_dict[self.data_key] = augment_gamma(data_dict[self.data_key], self.gamma_range, self.invert_image,
                                                 per_channel=self.per_channel, retain_stats=self.retain_stats,
                                                 p_per_sample=self.p_per_sample)
        return data_dict


def map_labels_to_trainId(arr):
    """Remap ids to corresponding training Ids. Note that the inplace mapping works because id > trainId here!"""
    id2trainId   = {label.id:label.trainId for label in cityscapes_labels_tuple}
    for id, trainId in id2trainId.items():
        arr[arr == id] = trainId
    return arr


class AddLossMask(AbstractTransform):
    """Splits one-hot segmentation into a segmentation array and a loss mask,
    where the loss mask needs to be encoded as the next available integer larger than the last segmentation labels.

    Args:
        classes (tuple of int): All the class labels that are in the dataset

        output_key (string): key to use for output of the one hot encoding. Default is 'seg' but that will override any
        other existing seg channels. Therefore you have the option to change that. BEWARE: Any non-'seg' segmentations
        will not be augmented anymore. Use this only at the very end of your pipeline!
    """

    def __init__(self, label2mask, output_key="loss_mask"):
        self.output_key = output_key
        self.label2mask = label2mask

    def __call__(self, **data_dict):
        seg = data_dict['seg']
        if seg is not None:
            data_dict[self.output_key] = (seg == self.label2mask).astype(np.uint8)
        else:
            from warnings import warn
            warn("calling AddLossMask but there is no segmentation")
        return data_dict


class StochasticLabelSwitches(AbstractTransform):
    """
    Stochastically switches labels in a batch of integer-labeled segmentations.
    """
    def __init__(self, name2id, label_switches):
        self._name2id = name2id
        self._label_switches = label_switches

    def __call__(self, **data_dict):

        switched_seg = data_dict['seg']
        batch_size = switched_seg.shape[0]

        for c, p in self._label_switches.items():
            init_id = self._name2id[c]
            final_id = self._name2id[c + '_2']
            switch_instances = np.random.binomial(1, p, batch_size)

            for i in range(batch_size):
                if switch_instances[i]:
                    switched_seg[i][switched_seg[i] == init_id] = final_id

        data_dict['seg'] = switched_seg
        return data_dict


class BatchGenerator(SlimDataLoaderBase):
    """
    create the training/validation batch generator. Randomly sample n_batch_size patients
    from the data set, (draw a random slice if 2D), pad-crop them to equal sizes and merge to an array.
    :param data: data dictionary as provided by 'load_dataset'
    :param batch_size: number of patients to sample for the batch
    :param pre_crop_size: equal size for merging the patients to a single array (before the final random-crop in data aug.)
    :return dictionary containing the batch data / seg / pids
    """
    def __init__(self, batch_size, data_dir, label_density='gtFine', data_split='train', resolution='quarter',
                 cities=None, gt_instances=False, n_batches=None, random=True):
        super(BatchGenerator, self).__init__(data=None, batch_size=batch_size)

        data_dir = os.path.join(data_dir, resolution)
        self._data_dir = data_dir
        self._label_density = label_density
        self._gt_instances = gt_instances
        self._data_split = data_split
        self._random = random
        self._n_batches = n_batches
        self._batches_generated = 0
        self._data = loadFiles(label_density, data_split, data_dir, cities=cities, instance=gt_instances)
        logging.info('{} set comprises {} files.'.format(data_split, len(self._data)))

    def generate_train_batch(self):

        if self._random:
            img_ixs = np.random.choice(list(self._data.keys()), self.batch_size, replace=True)
        else:
            batch_no = self._batches_generated % self._n_batches
            img_ixs = [list(self._data.keys())[i] for i in\
                       np.arange(batch_no * self.batch_size, (batch_no + 1) * self.batch_size)]
        img_batch, seg_batch, ids_batch = [], [], []

        for b in range(self.batch_size):

            img = np.load(self._data[img_ixs[b]]['data']) / 255.
            seg = np.load(self._data[img_ixs[b]]['seg'])
            seg = map_labels_to_trainId(seg)
            seg = seg[np.newaxis]
            ids_batch.append(self._data[img_ixs[b]]['img_spec'])

            img_batch.append(img)
            seg_batch.append(seg)

        self._batches_generated += 1
        batch = {'data': np.array(img_batch).astype('float32'), 'seg': np.array(seg_batch).astype('uint8'),
                 'id': ids_batch}
        return batch


def create_data_gen_pipeline(cf, cities=None, data_split='train', do_aug=True, random=True, n_batches=None):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param cities: list of strings or None
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset
    :param test_pids: (optional) list of test patient ids, calls the test generator.
    :param do_aug: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :param random: bool, whether to draw random batches or go through data linearly
    :return: multithreaded_generator
    """
    data_gen = BatchGenerator(cities=cities, batch_size=cf.batch_size, data_dir=cf.data_dir,
                              label_density=cf.label_density, data_split=data_split, resolution=cf.resolution,
                              gt_instances=cf.gt_instances, n_batches=n_batches, random=random)
    my_transforms = []
    if do_aug:
        mirror_transform = MirrorTransform(axes=(3,))
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[-2:],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'],
                                             border_mode_data=cf.da_kwargs['border_mode_data'],
                                             border_mode_seg=cf.da_kwargs['border_mode_seg'],
                                             border_cval_seg=cf.da_kwargs['border_cval_seg'])
        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[-2:]))

    my_transforms.append(GammaTransform(cf.da_kwargs['gamma_range'], invert_image=False, per_channel=True,
                                        retain_stats=cf.da_kwargs['gamma_retain_stats'],
                                        p_per_sample=cf.da_kwargs['p_gamma']))
    my_transforms.append(AddLossMask(cf.ignore_label))
    if cf.label_switches is not None:
        my_transforms.append(StochasticLabelSwitches(cf.name2trainId, cf.label_switches))
    all_transforms = Compose(my_transforms)
    multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers,
                                                     seeds=range(cf.n_workers))
    return multithreaded_generator


def get_train_generators(cf):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators
    """
    batch_gen = {}
    batch_gen['train'] = create_data_gen_pipeline(cf=cf, cities=cf.train_cities, data_split='train', do_aug=True,
                                                  n_batches=cf.n_train_batches)
    batch_gen['val'] = create_data_gen_pipeline(cf=cf, cities=cf.val_cities, data_split='train', do_aug=False,
                                                random=False, n_batches=cf.n_val_batches)
    return batch_gen

def main():
    """Main entry point for the script."""
    logging.info("start loading.")
    cf = imp.load_source('cf', 'config.py')
    dict = loadFiles("gtFine", "train", cf.out_dir, False)
    logging.info('Contains {} elements.'.format(len(dict)))
    logging.info(dict)
    data_provider = BatchGenerator(8, cf.out_dir, data_split='val')
    batch = next(data_provider)
    logging.info(batch['data'].shape, batch['seg'].shape, batch['id'])


if __name__ == '__main__':
    sys.exit(main())

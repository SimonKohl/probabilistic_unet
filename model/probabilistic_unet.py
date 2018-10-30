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
"""Probabilistic U-Net model."""

import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
tfd = tfp.distributions
from utils.training_utils import ce_loss, he_normal

def down_block(features,
               output_channels,
               kernel_shape,
               stride=1,
               rate=1,
               num_convs=2,
               initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
               regularizers=None,
               nonlinearity=tf.nn.relu,
               down_sample_input=True,
               down_sampling_op=lambda x, df: tf.nn.avg_pool(x, ksize=[1,1,2,2], strides=[1,1,2,2],
                                                padding='SAME', data_format=df),
               data_format='NCHW',
               name='down_block'):
    """A block made up of a down-sampling step followed by several convolutional layers."""
    with tf.variable_scope(name):
        if down_sample_input:
            features = down_sampling_op(features, data_format)

        for _ in range(num_convs):
            features = snt.Conv2D(output_channels, kernel_shape, stride, rate, data_format=data_format,
                                  initializers=initializers, regularizers=regularizers)(features)
            features = nonlinearity(features)

        return features


def up_block(lower_res_inputs,
             same_res_inputs,
             output_channels,
             kernel_shape,
             stride=1,
             rate=1,
             num_convs=2,
             initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
             regularizers=None,
             nonlinearity=tf.nn.relu,
             up_sampling_op=lambda x, size: tf.image.resize_images(x, size,
                                            method=tf.image.ResizeMethod.BILINEAR, align_corners=True),
             data_format='NCHW',
             name='up_block'):
    """A block made up of an up-sampling step followed by several convolutional layers."""
    with tf.variable_scope(name):
        spatial_shape = same_res_inputs.get_shape()[2:]

        if data_format=='NHWC':
            features = up_sampling_op(lower_res_inputs, spatial_shape)
            features = tf.concat([features, same_res_inputs], axis=-1)
        else:
            lower_res_inputs = tf.transpose(lower_res_inputs, perm=[0,2,3,1])
            features = up_sampling_op(lower_res_inputs, spatial_shape)
            features = tf.transpose(features, perm=[0,3,1,2])
            features = tf.concat([features, same_res_inputs], axis=1)

        for _ in range(num_convs):
            features = snt.Conv2D(output_channels, kernel_shape, stride, rate, data_format=data_format,
                                  initializers=initializers, regularizers=regularizers)(features)
            features = nonlinearity(features)

        return features


class VGG_Encoder(snt.AbstractModule):
    """A quasi VGG-style convolutional net, made of M x (down-sampling, N x conv)-operations,
       where M = len(num_channels), N = num_convs_per_block."""

    def __init__(self,
                 num_channels,
                 nonlinearity=tf.nn.relu,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 down_sampling_op=lambda x, df: tf.nn.avg_pool(x, ksize=[1,1,2,2], strides=[1,1,2,2],
                                                padding='SAME', data_format=df),
                 name="vgg_enc"):
        super(VGG_Encoder, self).__init__(name=name)
        self._num_channels = num_channels
        self._nonlinearity = nonlinearity
        self._num_convs = num_convs_per_block
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format
        self._down_sampling_op = down_sampling_op

    def _build(self, inputs):
        """
        :param inputs: 4D tensor of shape NCHW or NWHC
        :return: a list of 4D tensors of shape NCHW or NWHC
        """
        features = [inputs]

        # iterate blocks (`processing scales')
        for i, n_channels in enumerate(self._num_channels):

            if i == 0:
                down_sample = False
            else:
                down_sample = True
            tf.logging.info('encoder scale {}: {}'.format(i, features[-1].get_shape()))
            features.append(down_block(features[-1],
                                       output_channels=n_channels,
                                       kernel_shape=(3,3),
                                       num_convs=self._num_convs,
                                       nonlinearity=self._nonlinearity,
                                       initializers=self._initializers,
                                       regularizers=self._regularizers,
                                       down_sample_input=down_sample,
                                       data_format=self._data_format,
                                       down_sampling_op=self._down_sampling_op,
                                       name='down_block_{}'.format(i)))
        # return all features except for the input images
        return features[1:]


class VGG_Decoder(snt.AbstractModule):
    """A quasi VGG-style convolutional net, made of M x (up-sampling, N x conv)-operations,
       where M = len(num_channels), N = num_convs_per_block."""

    def __init__(self,
                 num_channels,
                 num_classes,
                 nonlinearity=tf.nn.relu,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 up_sampling_op=lambda x, size: tf.image.resize_images(x, size,
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True),
                 name="vgg_dec"):
        super(VGG_Decoder, self).__init__(name=name)
        self._num_channels = num_channels
        self._num_classes = num_classes
        self._nonlinearity = nonlinearity
        self._num_convs = num_convs_per_block
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format
        self._up_sampling_op = up_sampling_op

    def _build(self, input_list):
        """
        :param input_list: a list of 4D tensors of shape NCHW or NHWC
        :return: 4D tensor
        """
        try:
            assert len(self._num_channels) == len(input_list)
        except:
            raise AssertionError('Missmatch: {} blocks vs. {} incoming feature-maps!')

        n = len(input_list) - 2
        lower_res_features = input_list[-1]

        # iterate input features & channels in reverse order, starting from the second last (here, =nth) features
        for i in range(n, -1, -1):
            same_res_features = input_list[i]
            n_channels = self._num_channels[i]

            tf.logging.info('decoder scale {}: {}'.format(i, lower_res_features.get_shape()))
            lower_res_features = up_block(lower_res_features,
                                          same_res_features,
                                          output_channels=n_channels,
                                          kernel_shape=(3, 3),
                                          num_convs=self._num_convs,
                                          nonlinearity=self._nonlinearity,
                                          initializers=self._initializers,
                                          regularizers=self._regularizers,
                                          data_format=self._data_format,
                                          up_sampling_op=self._up_sampling_op,
                                          name='up_block_{}'.format(i))
        return lower_res_features


class UNet(snt.AbstractModule):
    """A quasi standard U-Net, similar to `U-Net: Convolutional Networks for Biomedical Image Segmentation',
     https://arxiv.org/abs/1505.04597."""

    def __init__(self,
                 num_channels,
                 num_classes,
                 nonlinearity=tf.nn.relu,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers=None,
                 data_format='NCHW',
                 down_sampling_op=lambda x, df: tf.nn.avg_pool(x, ksize=[1,1,2,2], strides=[1,1,2,2],
                                                padding='SAME', data_format=df),
                 up_sampling_op=lambda x, size: tf.image.resize_images(x, size,
                                                method=tf.image.ResizeMethod.BILINEAR, align_corners=True),
                 name="unet"):
        super(UNet, self).__init__(name=name)
        with self._enter_variable_scope():
            tf.logging.info('Building U-Net.')
            self._encoder = VGG_Encoder(num_channels, nonlinearity, num_convs_per_block, initializers, regularizers,
                                        data_format=data_format, down_sampling_op=down_sampling_op)
            self._decoder = VGG_Decoder(num_channels, num_classes, nonlinearity, num_convs_per_block, initializers,
                                        regularizers, data_format=data_format, up_sampling_op=up_sampling_op)

    def _build(self, inputs):
        """
        :param inputs: 4D tensor of shape NCHW or NHWC
        :return: 4D tensor
        """
        encoder_features = self._encoder(inputs)
        predicted_logits = self._decoder(encoder_features)
        return predicted_logits


class Conv1x1Decoder(snt.AbstractModule):
    """A stack of 1x1 convolutions that takes two tensors to be concatenated along their channel axes."""

    def __init__(self,
                 num_classes,
                 num_channels,
                 num_1x1_convs,
                 nonlinearity=tf.nn.relu,
                 initializers={'w': tf.orthogonal_initializer(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 name='conv_decoder'):
        super(Conv1x1Decoder, self).__init__(name=name)
        self._num_classes = num_classes
        self._num_channels = num_channels
        self._num_1x1_convs = num_1x1_convs
        self._nonlinearity = nonlinearity
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format

        if data_format == 'NCHW':
            self._channel_axis = 1
            self._spatial_axes = [2,3]
        else:
            self._channel_axis = -1
            self._spatial_axes = [1,2]

    def _build(self, features, z):
        """
        :param features: 4D tensor of shape NCHW or NHWC
        :param z: 4D tensor of shape NC11 or N11C
        :return: 4D tensor
        """
        shp = features.get_shape()
        spatial_shape = [shp[axis] for axis in self._spatial_axes]
        multiples = [1] + spatial_shape
        multiples.insert(self._channel_axis, 1)

        if len(z.get_shape()) == 2:
            z = tf.expand_dims(z, axis=2)
            z = tf.expand_dims(z, axis=2)

        # broadcast latent vector to spatial dimensions of the image/feature tensor
        broadcast_z = tf.tile(z, multiples)
        features = tf.concat([features, broadcast_z], axis=self._channel_axis)
        for _ in range(self._num_1x1_convs):
            features = snt.Conv2D(self._num_channels, kernel_shape=(1,1), stride=1, rate=1,
                                  data_format=self._data_format,
                                  initializers=self._initializers, regularizers=self._regularizers)(features)
            features = self._nonlinearity(features)
        logits = snt.Conv2D(self._num_classes, kernel_shape=(1,1), stride=1, rate=1,
                           data_format=self._data_format,
                           initializers=self._initializers, regularizers=None)
        return logits(features)


class AxisAlignedConvGaussian(snt.AbstractModule):
    """A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix."""

    def __init__(self,
                 latent_dim,
                 num_channels,
                 nonlinearity=tf.nn.relu,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 down_sampling_op=lambda x, df:\
                         tf.nn.avg_pool(x, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME', data_format=df),
                 name="conv_dist"):
        self._latent_dim = latent_dim
        self._initializers = initializers
        self._regularizers = regularizers
        self._data_format = data_format

        if data_format == 'NCHW':
            self._channel_axis = 1
            self._spatial_axes = [2,3]
        else:
            self._channel_axis = -1
            self._spatial_axes = [1,2]

        super(AxisAlignedConvGaussian, self).__init__(name=name)
        with self._enter_variable_scope():
            tf.logging.info('Building ConvGaussian.')
            self._encoder = VGG_Encoder(num_channels, nonlinearity, num_convs_per_block, initializers, regularizers,
                                        data_format=data_format, down_sampling_op=down_sampling_op)

    def _build(self, img, seg=None):
        """
        Evaluate mu and log_sigma of a Gaussian conditioned on an image + optionally, concatenated one-hot segmentation.
        :param img: 4D array
        :param seg: 4D array
        :return: snt.AbstractModule object
        """
        if seg is not None:
            seg = tf.cast(seg, tf.float32)
            img = tf.concat([img, seg], axis=self._channel_axis)
        encoding = self._encoder(img)[-1]
        encoding = tf.reduce_mean(encoding, axis=self._spatial_axes, keepdims=True)

        mu_log_sigma = snt.Conv2D(2 * self._latent_dim, (1,1), stride=1, rate=1, data_format=self._data_format,
                                 initializers=self._initializers, regularizers=self._regularizers)(encoding)

        mu_log_sigma = tf.squeeze(mu_log_sigma, axis=self._spatial_axes)
        mu = mu_log_sigma[:, :self._latent_dim]
        log_sigma = mu_log_sigma[:, self._latent_dim:]

        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_sigma))


class ProbUNet(snt.AbstractModule):
    """Probabilistic U-Net."""

    def __init__(self,
                 latent_dim,
                 num_channels,
                 num_classes,
                 num_1x1_convs=3,
                 nonlinearity=tf.nn.relu,
                 num_convs_per_block=3,
                 initializers={'w': he_normal(), 'b': tf.truncated_normal_initializer(stddev=0.001)},
                 regularizers={'w': tf.contrib.layers.l2_regularizer(1.0), 'b': tf.contrib.layers.l2_regularizer(1.0)},
                 data_format='NCHW',
                 down_sampling_op=lambda x, df:\
                         tf.nn.avg_pool(x, ksize=[1,1,2,2], strides=[1,1,2,2], padding='SAME', data_format=df),
                 up_sampling_op=lambda x, size:\
                         tf.image.resize_images(x, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=True),
                 name='prob_unet'):
        super(ProbUNet, self).__init__(name=name)
        self._data_format = data_format
        self._num_classes = num_classes

        with self._enter_variable_scope():
            self._unet = UNet(num_channels=num_channels, num_classes=num_classes, nonlinearity=nonlinearity,
                              num_convs_per_block=num_convs_per_block, initializers=initializers,
                              regularizers=regularizers, data_format=data_format,
                              down_sampling_op=down_sampling_op, up_sampling_op=up_sampling_op)

            self._f_comb = Conv1x1Decoder(num_classes=num_classes, num_1x1_convs=num_1x1_convs,
                                          num_channels=num_channels[0], nonlinearity=nonlinearity,
                                          data_format=data_format, initializers=initializers, regularizers=regularizers)

            self._prior =\
                AxisAlignedConvGaussian(latent_dim=latent_dim, num_channels=num_channels,
                                        nonlinearity=nonlinearity, num_convs_per_block=num_convs_per_block,
                                        initializers=initializers, regularizers=regularizers, name='prior')

            self._posterior =\
                AxisAlignedConvGaussian(latent_dim=latent_dim, num_channels=num_channels,
                                        nonlinearity=nonlinearity, num_convs_per_block=num_convs_per_block,
                                        initializers=initializers, regularizers=regularizers, name='posterior')

    def _build(self, img, seg=None, is_training=True, one_hot_labels=True):
        """
        Evaluate individual components of the net.
        :param img: 4D image array
        :param seg: 4D segmentation array
        :param is_training: if False, refrain from evaluating the posterior
        :param one_hot_labels: bool, if False expects integer labeled segmentation of shape N1HW or NHW1
        :return: None
        """
        if is_training:
            if seg is not None:
                if not one_hot_labels:
                    if self._data_format == 'NCHW':
                        spatial_shape = img.get_shape()[-2:]
                        class_axis = 1
                        one_hot_shape = (-1, self._num_classes) + tuple(spatial_shape)
                    elif self._data_format == 'NHWC':
                        spatial_shape = img.get_shape()[-3:-1]
                        one_hot_shape = (-1,) + tuple(spatial_shape) + (self._num_classes,)
                        class_axis = 3

                    seg = tf.reshape(seg, shape=[-1])
                    seg = tf.one_hot(indices=seg, depth=self._num_classes, axis=class_axis)
                    seg = tf.reshape(seg, shape=one_hot_shape)
                seg -= 0.5
            self._q = self._posterior(img, seg)

        self._p = self._prior(img)
        self._unet_features = self._unet(img)

    def reconstruct(self, use_posterior_mean=False, z_q=None):
        """
        Reconstruct a given segmentation. Default settings result in decoding a posterior sample.
        :param use_posterior_mean: use posterior_mean instead of sampling z_q
        :param z_q: use provided latent sample z_q instead of sampling anew
        :return: 4D logits tensor
        """
        if use_posterior_mean:
            z_q = self._q.loc
        else:
            if z_q is None:
                z_q = self._q.sample()
        return self._f_comb(self._unet_features, z_q)

    def sample(self):
        """
        Sample a segmentation by reconstructing from a prior sample.
        Only needs to re-evaluate the last 1x1-convolutions.
        :return: 4D logits tensor
        """
        z_p = self._p.sample()
        return self._f_comb(self._unet_features, z_p)

    def kl(self, analytic=True, z_q=None):
        """
        Calculate the Kullback-Leibler divergence KL(Q||P) between 2 axis-aligned gaussians,
        i.e. the variance sigma is assumed diagonal.
        :param analytic: bool, if False, approximate the KL via sampling from the posterior
        :param z_q: None or 2D tensor, if analytic=False the posterior sample can be provided instead of sampling anew
        :return: 4D tensor
        """
        if analytic:
            kl = tfd.kl_divergence(self._q, self._p)
        else:
            if z_q is None:
                z_q = self._q.sample()
            log_q = self._q.log_prob(z_q)
            log_p = self._p.log_prob(z_q)
            kl = log_q - log_p
        return kl

    def elbo(self, seg, beta=1.0, analytic_kl=True, reconstruct_posterior_mean=False, z_q=None, one_hot_labels=True,
             loss_mask=None):
        """
        Calculate the evidence lower bound (elbo) of the log-likelihood of P(Y|X).
        :param seg: 4D tensor
        :param analytic_kl: bool, if False calculate the KL via sampling
        :param z_q: 4D tensor
        :param one_hot_labels: bool, if False expects integer labeled segmentation of shape N1HW or NHW1
        :param loss_mask: 4D tensor, binary
        :return: 1D tensor
        """
        if z_q is None:
            z_q = self._q.sample()

        self._kl = tf.reduce_mean(self.kl(analytic_kl, z_q))

        self._rec_logits = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, z_q=z_q)
        rec_loss = ce_loss(labels=seg, logits=self._rec_logits, n_classes=self._num_classes,
                           loss_mask=loss_mask, one_hot_labels=one_hot_labels)
        self._rec_loss = rec_loss['sum']
        self._rec_loss_mean = rec_loss['mean']

        return -(self._rec_loss + beta * self._kl)
import tensorflow as tf
from .util import *
class Segmenter(object):
    def __init__(self, inputs, is_training, ochan, stddev=0.02, center=True, scale=True, reuse=None, attn=True):
        self._is_training = is_training
        self._stddev = stddev
        with tf.variable_scope('Segmentation', initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._inputs = inputs
            self._center = center
            self._scale = scale
            self._seg = self.unet_seg(inputs)

    def unet_seg(self, inputs, reuse=False, is_training=True):
        with tf.variable_scope('unet_seg', reuse=reuse):
            en1 = tf.layers.conv3d(inputs, 64, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            en1 = tf.layers.batch_normalization(en1, center=True, scale=True, training=is_training)
            en1 = tf.nn.relu(en1)
            pool1 = tf.layers.MaxPooling3D((2, 2, 2), strides=2, padding='same')(en1)

            en2 = tf.layers.conv3d(pool1, 128, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            en2 = tf.layers.batch_normalization(en2, center=True, scale=True, training=is_training)
            en2 = tf.nn.relu(en2)
            pool2 = tf.layers.MaxPooling3D((2, 2, 2), strides=2, padding='same')(en2)

            en3 = tf.layers.conv3d(pool2, 256, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            en3 = tf.layers.batch_normalization(en3, center=True, scale=True, training=is_training)
            en3 = tf.nn.relu(en3)
            pool3 = tf.layers.MaxPooling3D((2, 2, 2), strides=2, padding='same')(en3)

            en4 = tf.layers.conv3d(pool3, 512, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            en4 = tf.layers.batch_normalization(en4, center=True, scale=True, training=is_training)
            en4 = tf.nn.relu(en4)
            pool4 = tf.layers.MaxPooling3D((2, 2, 2), strides=2, padding='same')(en4)

            en5 = tf.layers.conv3d(pool4, 512, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            en5 = tf.layers.batch_normalization(en5, center=True, scale=True, training=is_training)
            en5 = tf.nn.relu(en5)
            #en5 = squeeze_and_excitation(en5, 512, 16, 'seg_se')

            layer = dict()
            de4 = tf.layers.conv3d_transpose(en5, 512, 4, (2, 2, 2), padding='same', data_format='channels_last', use_bias=None)
            concat4 = tf.concat([de4, en4], axis=4)
            de4 = tf.layers.conv3d(concat4, 512, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            de4 = tf.layers.batch_normalization(de4, center=True, scale=True, training=is_training)
            de4 = tf.nn.relu(de4)

            layer['out3'] = tf.layers.conv3d_transpose(de4, 8, 3, (8, 8, 8), padding='same', data_format='channels_last', use_bias=None)
            de3 = tf.layers.conv3d_transpose(de4, 256, 4, (2, 2, 2), padding='same', data_format='channels_last', use_bias=None)
            concat3 = tf.concat([de3, en3], axis=4)
            de3 = tf.layers.conv3d(concat3, 256, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            de3 = tf.layers.batch_normalization(de3, center=True, scale=True, training=is_training)
            de3 = tf.nn.relu(de3)

            layer['out2'] = tf.layers.conv3d_transpose(de3, 8, 3, (4, 4, 4), padding='same', data_format='channels_last', use_bias=None)
            de2 = tf.layers.conv3d_transpose(de3, 128, 4, (2, 2, 2), padding='same', data_format='channels_last', use_bias=None)
            concat2 = tf.concat([de2, en2], axis=4)
            de2 = tf.layers.conv3d(concat2, 128, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            de2 = tf.layers.batch_normalization(de2, center=True, scale=True, training=is_training)
            de2 = tf.nn.relu(de2)

            layer['out1'] = tf.layers.conv3d_transpose(de2, 8, 3, (2, 2, 2), padding='same', data_format='channels_last', use_bias=None)
            de1 = tf.layers.conv3d_transpose(de2, 64, 4, (2, 2, 2), padding='same', use_bias=None)
            concat1 = tf.concat([de1, en1], axis=4)
            de1 = tf.layers.conv3d(concat1, 64, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            de1 = tf.layers.batch_normalization(de1, center=True, scale=True, training=is_training)
            de1 = tf.nn.relu(de1)

            de0 = tf.layers.conv3d(de1, 8, 3, (1, 1, 1), padding='same', data_format='channels_last', use_bias=None)
            #out = tf.nn.sigmoid(de0)
            layer['out0'] = de0
            return layer


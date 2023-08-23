
from .util import *
import ipdb
# U-Net generator
class Generator(object):
    def __init__(self, inputs, is_training, ochan, stddev=0.02, center=True, scale=True, reuse=None, attn=True):
        self._is_training = is_training
        self._stddev = stddev
        self._ochan = ochan
        self.sigma_collection = dict()
        self.attn = attn
        with tf.variable_scope('Generator', initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._center = center
            self._scale = scale
            self._prob = 0.5  # constant from pix2pix paper
            self._inputs = inputs
            self._encoder = self._build_encoder(inputs)
            self._decoder = self._build_decoder(self._encoder)


    def _build_encoder_layer(self, name, size, inputs, k, bn=True, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name, size):
            layer['conv'] = snconv3d(inputs, size=size, output_dim=k,  stride=2)
            layer['bn'] = batch_norm(layer['conv'], center=self._center, scale=self._scale, training=self._is_training)\
                          if bn else layer['conv']
            #layer['relu'] = lkrelu(layer['bn'], slope=0.2)
            layer['relu'] = tf.nn.relu(layer['bn'])
            layer['fmap'] = tf.nn.dropout(layer['relu'], self._prob) if use_dropout else layer['relu']

            return layer

    def _build_bottlenek_layer(self, input,filter_name):
        layer = dict()
        w = tf.get_variable(filter_name, [4, 4, 4, get_shape(input)[-1], 512])
        layer['conv'] = tf.nn.conv3d(input, w, [1, 1, 1, 1, 1], padding='SAME')
        layer['bn'] = tf.layers.batch_normalization(layer['conv'], center=self._center, scale=self._scale, training=self._is_training)
        layer['relu'] = tf.nn.relu(layer['bn'])
        layer['fmap'] = tf.nn.dropout(layer['relu'], self._prob)
        layer['concat'] = tf.concat([layer['fmap'], input], axis=4)
        return layer

    def _build_encoder(self, inputs):
        encoder = dict()
        with tf.variable_scope('encoder'):
            encoder['l1'] = self._build_encoder_layer('l1', 5, inputs, 64, bn=False)
            encoder['l2'] = self._build_encoder_layer('l2', 5, encoder['l1']['fmap'], 128)
            encoder['l3'] = self._build_encoder_layer('l3', 3, encoder['l2']['fmap'], 256)
            encoder['l4'] = self._build_encoder_layer('l4', 3, encoder['l3']['fmap'], 512)
            if self.attn:
                encoder['encoder_attention'], encoder_sigma = sn_attention('encoder_attention', encoder['l4']['fmap'])
                self.sigma_collection['encoder_sigma'] = encoder_sigma
            else:
                encoder['encoder_attention'] = encoder['l4']['fmap']
            encoder['l5'] = self._build_encoder_layer('l5', 3, encoder['encoder_attention'], 512)#shape=(?, 4, 4, 4, 512)
            #encoder['encoder_se'] = squeeze_and_excitation(encoder['l5']['fmap'], 512, 16, 'encoder_se')
            encoder['l6'] = self._build_encoder_layer('l6', 3, encoder['l5']['fmap'], 512)
            #encoder['l6-1'] = self._build_bottlenek_layer(encoder['l6']['fmap'], filter_name='filter1')
            #encoder['l6-2'] = self._build_bottlenek_layer(encoder['l6-1']['concat'], filter_name='filter2')
            #encoder['l6-3'] = self._build_bottlenek_layer(encoder['l6-2']['concat'], filter_name='filter3')
        return encoder

    def _build_decoder_layer(self, name, inputs, size, stride, output_shape_from, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name):
            output_shape = tf.shape(output_shape_from)
            layer['conv'] = snconv3d_tranpose(inputs, size=size, output_dim_from=output_shape_from, stride=stride)
            layer['bn'] = batch_norm(tf.reshape(layer['conv'], output_shape), center=self._center, scale=self._scale,
                                     training=self._is_training)
            #layer['relu'] = lkrelu(layer['bn'], slope=0.2)
            layer['relu'] = tf.nn.relu(layer['bn'])
            layer['fmap'] = tf.nn.dropout(layer['relu'], self._prob) if use_dropout else layer['relu']

        return layer

    def _build_decoder(self, encoder):
        decoder = dict()
        with tf.variable_scope('decoder'):
            decoder['dl1'] = self._build_decoder_layer('dl1', encoder['l6']['fmap'], 3, 2, output_shape_from=encoder['l5']['fmap'],
                                                       use_dropout=True)
            fmap_concat = tf.concat([decoder['dl1']['fmap'], encoder['l5']['fmap']], axis=4)

            decoder['dl2'] = self._build_decoder_layer('dl2', fmap_concat, 3, 2, output_shape_from=encoder['l4']['fmap'],
                                                       use_dropout=True)
            fmap_concat = tf.concat([decoder['dl2']['fmap'], encoder['l4']['fmap']], axis=4)

            decoder['dl3'] = self._build_decoder_layer('dl3', fmap_concat, 3, 2, output_shape_from=encoder['l3']['fmap'],
                                                       use_dropout=True)
            fmap_concat = tf.concat([decoder['dl3']['fmap'], encoder['l3']['fmap']], axis=4)

            decoder['dl4'] = self._build_decoder_layer('dl4', fmap_concat, 3, 2, output_shape_from=encoder['l2']['fmap'])

            if self.attn:
                decoder['decoder_attention_inter'], _ = sn_attention('decoder_attention', decoder['dl4']['fmap'])
            else:
                decoder['decoder_attention_inter'] = decoder['dl4']['fmap']

            fmap_concat = tf.concat([decoder['decoder_attention_inter'], encoder['l2']['fmap']], axis=4)
            decoder['dl5'] = self._build_decoder_layer('dl5', fmap_concat, 5, 2, output_shape_from=encoder['l1']['fmap'])

            fmap_concat = tf.concat([decoder['dl5']['fmap'], encoder['l1']['fmap']], axis=4)
            decoder['dl6'] = self._build_decoder_layer('dl6', fmap_concat, 5, 2, output_shape_from=self._inputs)

            with tf.variable_scope('final'):
                cl7 = dict()
                cl7['conv'] = snconv3d(decoder['dl6']['fmap'], size=3, stride=1, output_dim=self._ochan)
                if self.attn:
                    cl7['decoder_attention'], decoder_sigma = sn_attention('decoder_attention', cl7['conv'], final_layer=True)
                    self.sigma_collection['decoder_sigma'] = decoder_sigma
                else:
                    cl7['decoder_attention'] = cl7['conv']
                #ipdb.set_trace()
                cl7['fmap'] = tf.math.tanh(cl7['decoder_attention'], name='tanh')
#Tensor("Generator/decoder/final/tanh:0", shape=(?, 128, 128, 128, 1), dtype=float32, device=/device:GPU:1)
                decoder['final'] = cl7

        return decoder

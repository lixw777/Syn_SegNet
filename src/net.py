import tensorflow as tf
from src.utils.discriminator import Discriminator
from src.utils.generator_7t1 import Generator
#from src.utils.segmenter import Segmenter
from src.utils.unet_seg import Segmenter
from src.utils.util import *


class SCGAN(object):
    def __init__(self, width, height, depth, ichan, ochan, g_reg=0.001, d_reg=0.001, l1_weight=200, B_weight=200, lr=0.0002, beta1=0.5, floss=False, reg=False, attn=True):
        self._is_training = tf.placeholder(tf.bool, name='is_train_holder')
        self._g_inputs_3t = tf.placeholder(tf.float32, [None, width, height, depth, ichan], name='input_holder')#3tt1 3tt2
        self._d_inputs_3t = tf.placeholder(tf.float32, [None, width, height, depth, ichan])#3tt1 3tt2
        self._s_inputs_3t = tf.placeholder(tf.float32, [None, width, height, depth, ichan])
        self._d_inputs_7t1 = tf.placeholder(tf.float32, [None, width, height, depth, ochan])#real 7tt1
        self._input_7t1_label = tf.placeholder(tf.int32, [None, width, height, depth, 8])
        self._seg_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._label1_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._label2_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._label3_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._label4_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._label5_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._label6_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._label7_mask = tf.placeholder(tf.float32, [None, width, height, depth, None])
        self._g_7t1 = Generator(self._g_inputs_3t, self._is_training, ochan, attn=attn)
        self.lr = lr

        #real 7tt1 syn 7tt1 discriminator
        self._real_d_7t1 = Discriminator('Discriminator_2', tf.concat([self._d_inputs_3t, self._d_inputs_7t1], axis=4), self._is_training, attn=True)
        self._fake_d_7t1 = Discriminator('Discriminator_2', tf.concat([self._d_inputs_3t, self._g_7t1._decoder['final']['fmap']], axis=4), self._is_training, reuse=True, attn=True)

        self._f_matching_loss_7t1 = tf.reduce_mean([tf.reduce_mean(tf.abs(self._real_d_7t1._perceptual_fmap[i] - self._fake_d_7t1._perceptual_fmap[i]))
                                 for i in range(len(self._real_d_7t1._perceptual_fmap))])

        self._pure_g_loss = -tf.reduce_mean(tf.log(self._fake_d_7t1._discriminator['l5']['fmap']))

        #7tt1
        #rmse_mask1 = tf.not_equal(self._d_inputs_7t1, -1)
        #mask1 = tf.dtypes.cast(rmse_mask1, tf.float32)
        rmse_mask = tf.dtypes.cast(self._seg_mask, tf.bool)
        masked_input1 = tf.math.multiply(self._d_inputs_7t1, self._seg_mask)
        masked_output1 = tf.math.multiply(self._g_7t1._decoder['final']['fmap'], self._seg_mask)

        self.psnr_score = tf.reduce_mean(tf.image.psnr(self._d_inputs_7t1, self._g_7t1._decoder['final']['fmap'], max_val=2.0))
        self.ssim_score = tf.reduce_mean(tf.image.ssim(masked_input1, masked_output1, max_val=2.0))
        self.rmse_score = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(tf.boolean_mask(self._d_inputs_7t1, rmse_mask),
                                                      tf.boolean_mask(self._g_7t1._decoder['final']['fmap'], rmse_mask))))
        #self.seg_ave_mask_loss = tf.reduce_mean(tf.abs(tf.math.multiply(self._d_inputs_7t1, self._seg_mask) - tf.math.multiply(self._g_7t1._decoder['final']['fmap'], self._seg_mask)))

        self._g_loss_7t = self._pure_g_loss + \
                          l1_weight * (tf.reduce_mean(tf.abs(self._d_inputs_7t1 - self._g_7t1._decoder['final']['fmap']))) + B_weight * self.rmse_score

        self._d_loss_7t1 = -tf.reduce_mean(
            tf.log(self._real_d_7t1._discriminator['l5']['fmap'] + tf.keras.backend.epsilon()) +
            tf.log(1.0 - self._fake_d_7t1._discriminator['l5']['fmap'] + tf.keras.backend.epsilon()))
        self._d_loss_7t = self._d_loss_7t1

        #segmentation
        #ipdb.set_trace()
        self._fake_seg_7t1 = Segmenter(tf.concat([self._g_7t1._decoder['final']['fmap'], self._s_inputs_3t], axis=4), self._is_training, ochan, attn=False)
        self._fake_seg_out_7t1 = self._fake_seg_7t1._seg['out0']
        self._fake_softmax_7t1 = tf.nn.softmax(self._fake_seg_out_7t1)
        #self._ce_loss = weighted_ce_loss(self._fake_seg_out_7t1, self._input_7t1_label)
        self._ce_loss_final = WACE_loss(self._fake_seg_out_7t1, self._input_7t1_label, self._label1_mask, self._label2_mask, self._label3_mask, self._label4_mask,
                                  self._label5_mask, self._label6_mask, self._label7_mask)
        self._ce_ds_loss3 = WACE_loss(self._fake_seg_7t1._seg['out3'], self._input_7t1_label, self._label1_mask, self._label2_mask, self._label3_mask, self._label4_mask,
                                  self._label5_mask, self._label6_mask, self._label7_mask)
        self._ce_ds_loss2 = WACE_loss(self._fake_seg_7t1._seg['out2'], self._input_7t1_label, self._label1_mask,self._label2_mask, self._label3_mask, self._label4_mask,
                                      self._label5_mask, self._label6_mask, self._label7_mask)
        self._ce_ds_loss1 = WACE_loss(self._fake_seg_7t1._seg['out1'], self._input_7t1_label, self._label1_mask,self._label2_mask, self._label3_mask, self._label4_mask,
                                      self._label5_mask, self._label6_mask, self._label7_mask)

        self._seg_loss = self._ce_loss_final + self._ce_ds_loss3 + self._ce_ds_loss2 + self._ce_ds_loss1

        self._g_loss_final = self._g_loss_7t
        self._d_loss_final = self._d_loss_7t
        self._f_matching_loss_2 = self._f_matching_loss_7t1
        if floss:
            self._f_matching_loss_2 = tf.cond(self._f_matching_loss_2 < 2e-4, lambda: 0.0, lambda: self._f_matching_loss_2)
            self._g_loss_final += 20 * self._f_matching_loss_2
            self.feature_matching_loss_2 = tf.summary.scalar("feature_matching_loss_2", self._f_matching_loss_2)

        if reg:
            vars = tf.trainable_variables(scope='Generator')
            self.l2_reg_g = tf.add_n([tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name]) * g_reg
            self.l2_reg_g = tf.cond(self.l2_reg_g < 1e-5, lambda: 0.0, lambda: self.l2_reg_g)
            self._g_loss_final += self.l2_reg_g
            tf.summary.scalar("generator_l2", self.l2_reg_g)

            vars = tf.trainable_variables(scope='Discriminator_2')
            self.l2_reg_d = tf.add_n([tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name]) * d_reg
            self.l2_reg_d = tf.cond(self.l2_reg_d < 5e-5, lambda: 0.0, lambda:self.l2_reg_d)
            self._d_loss_final += self.l2_reg_d
            tf.summary.scalar("discriminator_l2", self.l2_reg_d)

            vars = tf.trainable_variables(scope='Segmentation')
            self.l2_reg_s = tf.add_n([tf.nn.l2_loss(v) for v in vars
                                      if 'bias' not in v.name]) * d_reg
            self.l2_reg_s = tf.cond(self.l2_reg_s < 5e-5, lambda: 0.0, lambda: self.l2_reg_s)
            self._seg_loss += self.l2_reg_s
            tf.summary.scalar("discriminator_l2", self.l2_reg_s)

        tf.summary.scalar("pure generator_loss", self._pure_g_loss)
        tf.summary.scalar("generator_loss", self._g_loss_final)
        tf.summary.scalar('psnr score', self.psnr_score)
        tf.summary.scalar('ssim_score', self.ssim_score)
        tf.summary.scalar('rmse_score', self.rmse_score/2)
        tf.summary.scalar("discriminator_loss", -self._d_loss_final)

        self.l_rate = tf.placeholder(tf.float32, shape=None)

        #if attn:
            #tf.summary.scalar("decoder_sigma", self._g_7t1.sigma_collection['decoder_sigma'])
            #tf.summary.scalar("disc_fake_sigma", self._fake_d_2.sigma_collection['disc_sigma'])
        tf.summary.scalar("learning rate", self.l_rate)
        self.summary_merge = tf.summary.merge_all()

        # for test metrices
        self.test_psnr = tf.placeholder(tf.float32, shape=None)
        self.test_ssim = tf.placeholder(tf.float32, shape=None)
        self.test_rmse = tf.placeholder(tf.float32, shape=None)

        self.psnr = tf.summary.scalar('psnr test', self.test_psnr)
        self.ssim = tf.summary.scalar('ssim test', self.test_ssim)
        self.rmse = tf.summary.scalar('rmse test', self.test_rmse)

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
        with tf.control_dependencies(g_update_ops):
            self._g_train_step = tf.train.AdamOptimizer(self.lr, beta1=beta1)\
                                    .minimize(self._g_loss_final, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator_2')
        with tf.control_dependencies(d_update_ops):
            self._d_train_step_1 = tf.train.AdamOptimizer(self.lr, beta1=beta1)\
                                    .minimize(self._d_loss_final, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator_2'))

        s_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Segmentation')
        with tf.control_dependencies(s_update_ops):
            self._s_train_step_1 = tf.train.AdamOptimizer(self.lr) \
                                    .minimize(self._seg_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Segmentation'))

    def train_step(self, sess, g_inputs, d_inputs_a, d_inputs_7t1, inputs_7t1_label, inputs_seg_3t, seg_mask,label1_mask, label2_mask,label3_mask,label4_mask,label5_mask,label6_mask,label7_mask, is_training=True, train_d=1):

        _, gloss_curr = sess.run([self._g_train_step, self._g_loss_final],
                                 feed_dict={self._d_inputs_3t: d_inputs_a, self._d_inputs_7t1: d_inputs_7t1, self._seg_mask:seg_mask,
                                            self._g_inputs_3t: g_inputs, self._is_training: is_training})

        if train_d == 1:
            _, dloss_curr = sess.run([self._d_train_step_1, self._d_loss_final],
                                     feed_dict={self._d_inputs_3t: d_inputs_a, self._d_inputs_7t1: d_inputs_7t1, self._seg_mask:seg_mask,
                                                self._g_inputs_3t: g_inputs, self._is_training: is_training})
            _, sloss_curr = sess.run([self._s_train_step_1, self._seg_loss],
                                     feed_dict={self._d_inputs_3t: d_inputs_a, self._d_inputs_7t1: d_inputs_7t1, self._input_7t1_label: inputs_7t1_label, self._seg_mask:seg_mask,
                                                self._g_inputs_3t: g_inputs, self._s_inputs_3t: inputs_seg_3t, self._label1_mask:label1_mask,
                                                self._label2_mask:label2_mask,self._label3_mask:label3_mask,self._label4_mask:label4_mask,
                                                self._label5_mask:label5_mask,self._label6_mask:label6_mask,self._label7_mask:label7_mask,self._is_training: is_training})
            summart_temp = sess.run(self.summary_merge, feed_dict={self._d_inputs_3t: d_inputs_a, self._d_inputs_7t1: d_inputs_7t1, self._input_7t1_label: inputs_7t1_label,self._seg_mask:seg_mask,
                                            self._g_inputs_3t: g_inputs, self._s_inputs_3t: inputs_seg_3t, self._label1_mask:label1_mask,
                                                self._label2_mask:label2_mask,self._label3_mask:label3_mask,self._label4_mask:label4_mask,
                                                self._label5_mask:label5_mask,self._label6_mask:label6_mask,self._label7_mask:label7_mask,self._is_training: is_training, self.l_rate:self.lr})
            return gloss_curr, dloss_curr, sloss_curr, summart_temp

        summart_temp = sess.run(self.summary_merge, feed_dict={self._d_inputs_3t: d_inputs_a, self._d_inputs_7t1: d_inputs_7t1,self._input_7t1_label: inputs_7t1_label,
                                            self._g_inputs_3t: g_inputs, self._s_inputs_3t: inputs_seg_3t, self._is_training: is_training, self.l_rate:self.lr})

        return gloss_curr, _, summart_temp


    def sample_generator(self, sess, g_inputs, test_seg_3t,is_training=False):

        generated, segmented = sess.run([self._g_7t1._decoder['final']['fmap'], self._fake_softmax_7t1],
                            feed_dict={self._g_inputs_3t: g_inputs, self._s_inputs_3t: test_seg_3t, self._is_training: is_training})
        return generated, segmented

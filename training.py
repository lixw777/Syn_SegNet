
from src.utils.util import *
from src.net import SCGAN
from skimage.measure import compare_ssim
import os
import shutil
import nibabel as nib
import numpy as np
import tensorflow as tf
import copy
import ipdb
import keras
import faulthandler
faulthandler.enable()
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_float('g_reg', 0.001, 'generator regularizer')
flags.DEFINE_float('d_reg', 0.001, 'discriminator regularizer')
flags.DEFINE_boolean('floss', True, 'if using feature matching loss')
flags.DEFINE_boolean('reg', True, 'if using regularization')
flags.DEFINE_boolean('attn', True, 'if using self attention')
flags.DEFINE_integer('gpu', 2, 'the ID of gpu to use')
flags.DEFINE_integer('training_size', 15, 'training data size')
flags.DEFINE_integer('epoches', 1000, 'number of iteration')
flags.DEFINE_integer('l1_weight', 200, 'l1 weight')
flags.DEFINE_integer('B_weight', 200, 'B-rmse weight')
flags.DEFINE_integer('img_size', 128, 'image size')
flags.DEFINE_string('data_dir', '/home/student9/SC-GAN/data', 'training data directory')
flags.DEFINE_string('modalities', '3T1_3T2_7t1', 'modalities to use in the training')
flags.DEFINE_string('logdir', '/home/student9/SC-GAN-original/logs', 'path of tensorboard log')
flags.DEFINE_string('fold', 'fold2', 'which fold to use')

if FLAGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
def main(args):
    fold = FLAGS.fold
    if fold == 'fold1':
        test_set = ['01', '02', '03', '04']
        val_set = ['06', '09']
    if fold == 'fold2':
        test_set = ['05', '06', '07', '08']
        val_set = ['13', '15']
    if fold == 'fold3':
        test_set = ['09', '10', '11', '12']
        val_set = ['01', '20']
    if fold == 'fold4':
        test_set = ['13', '14', '15', '16']
        val_set = ['04', '19']
    if fold == 'fold5':
        test_set = ['17', '18', '19', '20']
        val_set = ['08', '16']
    all_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                '18', '19', '20']
    train_set = list(set(all_list) - set(test_set) - set(val_set))
    data_path = FLAGS.data_dir
    modalities = FLAGS.modalities.split('_')
    print('modalities:', modalities, modalities[:2], modalities[-1], len(modalities))
    num_samples = len(train_set)
    d_flag = 1
    with tf.device('/gpu:%s' %FLAGS.gpu):
        model = SCGAN(128, 128, 128, ichan=len(modalities)-1, ochan=1,
                      l1_weight=FLAGS.l1_weight, B_weight=FLAGS.B_weight, floss=FLAGS.floss, lr=FLAGS.lr,
                      reg=FLAGS.reg, attn=FLAGS.attn, g_reg=FLAGS.g_reg, d_reg=FLAGS.d_reg)

    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(tf.global_variables_initializer())
        dir = FLAGS.logdir
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        f_summary = tf.summary.FileWriter(logdir=dir, graph=sess.graph)
        previous_label_dice_ave = 0
        for step in range(1, FLAGS.epoches*num_samples+1):
            pos = step % num_samples
            if pos != 0:
                np.random.shuffle(train_set)
                subject = train_set[pos]
                input_list = list()
                for num_modalities in modalities[:2]:
                    input_3t = os.path.join(data_path, subject, '%s_to_chunktemp_left.nii.gz' % num_modalities)
                    img = nib.load(input_3t).get_fdata()
                    img = img / 100.
                    input_3t = np.zeros((128, 128, 128))
                    input_3t[:, :img.shape[1], :img.shape[2]] = img[:128, :, :]
                    input_3t = np.expand_dims(input_3t, axis=0)
                    input_3t = np.expand_dims(input_3t, axis=-1)
                    input_list.append(input_3t)
                input_3t = np.concatenate(input_list, axis=4)
                input_seg_3t = 2 * input_3t - 1

                input_7t1 = os.path.join(data_path, subject, '%s_to_chunktemp_left.nii.gz' % modalities[-1])
                img = nib.load(input_7t1).get_fdata()
                img = img / 100.
                input_7t1 = np.zeros((128, 128, 128))
                input_7t1[:, :img.shape[1], :img.shape[2]] = img[:128, :, :]
                input_7t1 = np.expand_dims(input_7t1, axis=0)
                input_7t1 = np.expand_dims(input_7t1, axis=-1)
                input_7t1 = 2 * input_7t1 - 1

                input_7t1_label = os.path.join(data_path, subject, 'seg-chunktemp-%s-left.nii.gz' % subject)
                img = nib.load(input_7t1_label).get_fdata()
                input_7t1_label = np.zeros((128, 128, 128))
                input_7t1_label[:, :img.shape[1], :img.shape[2]] = img[:128, :, :]
                input_7t1_label = np.expand_dims(input_7t1_label, axis=0)
                input_7t1_label = keras.utils.to_categorical(input_7t1_label,
                                       num_classes=8).reshape([input_7t1_label.shape[0], input_7t1_label.shape[1], input_7t1_label.shape[2], input_7t1_label.shape[3], 8])

                seg_mask = os.path.join(data_path, 'seg_ave_mask_final.nii.gz')
                seg_mask = nib.load(seg_mask).get_fdata()
                seg_mask = np.expand_dims(seg_mask, axis=0)
                seg_mask = np.expand_dims(seg_mask, axis=4)

                label1_mask = os.path.join(data_path, 'label_mask/seg_label1_mask_G.nii.gz')
                label1_mask = nib.load(label1_mask).get_fdata()
                label1_mask = np.expand_dims(label1_mask, axis=0)
                label1_mask = np.expand_dims(label1_mask, axis=4)

                label2_mask = os.path.join(data_path, 'label_mask/seg_label2_mask_G.nii.gz')
                label2_mask = nib.load(label2_mask).get_fdata()
                label2_mask = np.expand_dims(label2_mask, axis=0)
                label2_mask = np.expand_dims(label2_mask, axis=4)

                label3_mask = os.path.join(data_path, 'label_mask/seg_label3_mask_G.nii.gz')
                label3_mask = nib.load(label3_mask).get_fdata()
                label3_mask = np.expand_dims(label3_mask, axis=0)
                label3_mask = np.expand_dims(label3_mask, axis=4)

                label4_mask = os.path.join(data_path, 'label_mask/seg_label4_mask_G.nii.gz')
                label4_mask = nib.load(label4_mask).get_fdata()
                label4_mask = np.expand_dims(label4_mask, axis=0)
                label4_mask = np.expand_dims(label4_mask, axis=4)

                label5_mask = os.path.join(data_path, 'label_mask/seg_label5_mask_G.nii.gz')
                label5_mask = nib.load(label5_mask).get_fdata()
                label5_mask = np.expand_dims(label5_mask, axis=0)
                label5_mask = np.expand_dims(label5_mask, axis=4)

                label6_mask = os.path.join(data_path, 'label_mask/seg_label6_mask_G.nii.gz')
                label6_mask = nib.load(label6_mask).get_fdata()
                label6_mask = np.expand_dims(label6_mask, axis=0)
                label6_mask = np.expand_dims(label6_mask, axis=4)

                label7_mask = os.path.join(data_path, 'label_mask/seg_label7_mask_G.nii.gz')
                label7_mask = nib.load(label7_mask).get_fdata()
                label7_mask = np.expand_dims(label7_mask, axis=0)
                label7_mask = np.expand_dims(label7_mask, axis=4)

                model.lr = cosine_decay(learning_rate=FLAGS.lr, global_step=step, decay_steps=num_samples * 10, alpha=1e-10)
                gloss_curr, dloss_curr, sloss_curr, summary_temp = model.train_step(sess, input_3t, input_3t, input_7t1, input_7t1_label, input_seg_3t, seg_mask,
                                                                                    label1_mask, label2_mask,label3_mask,label4_mask,label5_mask,label6_mask,label7_mask,train_d=d_flag)
                f_summary.add_summary(summary=summary_temp, global_step=step)

                if d_flag == 1:
                    print('Step %d: generator loss: %f | discriminator loss1: %f | segmentation loss: %f ' %
                          (step, gloss_curr, dloss_curr, sloss_curr) + ' | ' + subject)
                else:
                    print('Step %d: generator loss: %f' % (step, gloss_curr))

            else:
                avg_psnr = []
                avg_ssim = []
                avg_rmse = []
                DSC_total = []
                #ipdb.set_trace()
                for val_subject in val_set:
                    val_list = list()
                    for num_modalities in modalities[:2]:
                        A_test = os.path.join(data_path, val_subject, '%s_to_chunktemp_left.nii.gz' % num_modalities)
                        img = nib.load(A_test).get_fdata()
                        img = img / 100.
                        A_test = np.zeros((128, 128, 128))
                        A_test[:, :img.shape[1], :img.shape[2]] = img[:128, :, :]
                        A_test = np.expand_dims(A_test, axis=0)
                        A_test = np.expand_dims(A_test, axis=-1)
                        val_list.append(A_test)
                    test_input = np.concatenate(val_list, axis=4)
                    test_seg_3t = test_input * 2. - 1

                    test_7t1 = os.path.join(data_path, val_subject, '%s_to_chunktemp_left.nii.gz' % modalities[-1])
                    test_7t1 = nib.load(test_7t1)
                    img = test_7t1.get_fdata()
                    img = img / 100.
                    test_7t1 = np.zeros((128, 128, 128))
                    test_7t1[:, :img.shape[1], :img.shape[2]] = img[:128, :, :]
                    temp_7t1 = test_7t1 * 2 - 1

                    test_7t1_label = os.path.join(data_path, val_subject, 'seg-chunktemp-%s-left.nii.gz' % val_subject)
                    img_affine = nib.load(test_7t1_label).affine
                    test_7t1_label = nib.load(test_7t1_label).get_fdata()
                    img = np.asarray(test_7t1_label)
                    img_label = np.zeros((128, 128, 128))
                    img_label[:, :img.shape[1], :img.shape[2]] = img[:128, :, :]

                    output, seg_out = model.sample_generator(sess, test_input, test_seg_3t, is_training=False)#output shape=(1,128, 128, 128,1)
                    output1 = output[0, :, :, :, 0]

                    seg_out = np.asarray(seg_out)
                    x_pred1 = np.argmax(seg_out, axis=4)
                    x_pred = np.squeeze(x_pred1)

                    dice = dice_eval(x_pred, img_label)
                    DSC_total.append(dice)
                    print(val_subject, dice)

                    img[:128, :, :] = x_pred[:, :img.shape[1], :img.shape[2]]

                    seg = nib.Nifti1Image(img, img_affine)
                    nib.save(seg, '/home/student9/SC-GAN-original/seg_%s.nii.gz' % val_subject)

                    mask1, masked_temp_pet1, masked_output1 = generate_mask(temp_7t1, output1)
                    dif = (masked_output1 - masked_temp_pet1)**2
                    mse_raw = []
                    mse2_raw = []
                    mse3_raw = []
                    for po in range(FLAGS.img_size):
                        mse_raw.append(dif[po,:,:][mask1[po,:,:]])
                        mse2_raw.append(dif[:,po,:][mask1[:,po,:]])
                        mse3_raw.append(dif[:,:,po][mask1[:,:,po]])

                    mse = get_mean(mse_raw)
                    mse2 = get_mean(mse2_raw)
                    mse3 = get_mean(mse3_raw)
                    rmse = (np.mean(np.sqrt(mse)) + np.mean(np.sqrt(mse2)) + np.mean(np.sqrt(mse3)))/3/2
                    mse = get_psnr_mean(mse_raw, FLAGS.img_size)
                    mse2 = get_psnr_mean(mse2_raw, FLAGS.img_size)
                    mse3 = get_psnr_mean(mse3_raw, FLAGS.img_size)
                    mse[mse == 0] = 1e-10
                    mse2[mse2 == 0] = 1e-10
                    mse3[mse3 == 0] = 1e-10
                    print(val_subject)
                    print('nrmse = %s' % rmse)
                    psnr = np.mean(10 * np.log10(4 / mse))
                    psnr2 = np.mean(10 * np.log10(4 / mse2))
                    psnr3 = np.mean(10 * np.log10(4 / mse3))
                    psnr = np.mean([psnr, psnr2, psnr3])
                    print('psnr = %s' % psnr)
                    ssim = []
                    for image in range(output1.shape[0]):
                        ssim.append(compare_ssim(masked_output1[image, :, :], masked_temp_pet1[image, :, :], data_range=2, win_size=11,
                                                gaussian_weights=True))
                    ssim = np.mean(ssim)
                    print('ssim = %s' % ssim)

                    avg_psnr.append(psnr)
                    avg_rmse.append(rmse)
                    avg_ssim.append(ssim)

                psnr = np.mean(avg_psnr)
                ssim = np.mean(avg_ssim)
                rmse = np.mean(avg_rmse)
                dice_ave = np.mean(np.asarray(DSC_total), axis=0)
                dice_ave = np.squeeze(dice_ave)
                label_dice_ave = (dice_ave[1]+dice_ave[2]+dice_ave[3]+dice_ave[4]+dice_ave[5]+dice_ave[6]+dice_ave[7])/7
                #if step != 0 and step % 3000 == 0:
                    #saver.save(sess, os.path.join(FLAGS.logdir, 'model_step_%s'%step))
                if label_dice_ave > previous_label_dice_ave:
                    saver.save(sess, os.path.join(FLAGS.logdir, 'model_step_%s' % step))
                    previous_label_dice_ave = label_dice_ave
                    print("saved step %s model" % step)
                    print("current best average dice%s" % label_dice_ave)
                """if ssim > previous_ssim:
                    saver.save(sess, os.path.join(FLAGS.logdir, 'model_step_%s' % step))#save_debug_info=True
                    previous_ssim = ssim
                    print("saved step %s model" % step)
                    print("current best ssim%s" % ssim)"""

                merge = tf.summary.merge([model.psnr, model.ssim, model.rmse])
                summary = sess.run(merge, feed_dict={model.test_psnr: psnr, model.test_ssim: ssim, model.test_rmse: rmse})
                f_summary.add_summary(summary=summary, global_step=step)

if __name__ == "__main__":
    tf.app.run()

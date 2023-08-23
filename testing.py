from src.utils.util import *
from skimage.measure import compare_ssim
import os
import nibabel as nib
import copy


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()

    keys_list = [keys for keys in flags_dict]
    print(keys_list)
    keys = 'log_dir'
    FLAGS.__delattr__(keys)
del_all_flags(tf.app.flags.FLAGS)
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('gpu', 2, 'the ID of gpu to use')
flags.DEFINE_integer('test_size', 2, 'test data size')
flags.DEFINE_integer('img_size', 128, 'image size')
flags.DEFINE_string('output', '/home/student9/SC-GAN-original/output', 'path of output folder')
flags.DEFINE_string('data_dir', '/home/student9/SC-GAN/data', 'test data directory')
flags.DEFINE_string('modalities', '3T1_3T2_7t1', 'modalities to use in the training')
flags.DEFINE_string('log_dir', '/home/student9/SC-GAN-original/logs', 'path to the tensorboard log')
flags.DEFINE_string('fold', 'fold2', 'which fold to use')
if FLAGS.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = '/gpu:%s'%FLAGS.gpu
modalities = FLAGS.modalities.split('_')

def main(args):
    fold = FLAGS.fold
    data_path = FLAGS.data_dir
    if fold == 'fold1':
        test_set = ['01', '02', '03', '04']
    if fold == 'fold2':
        test_set = ['05', '06', '07', '08']
    if fold == 'fold3':
        test_set = ['09', '10', '11', '12']
    if fold == 'fold4':
        test_set = ['13', '14', '15', '16']
    if fold == 'fold5':
        test_set = ['17', '18', '19', '20']
    log_list = os.listdir(FLAGS.log_dir)
    meta = []
    for name in log_list:
        if '.meta' in name:
            meta.append(name)
    meta.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
    meta_graph = os.path.join(FLAGS.log_dir, meta[-1])
    meta_weight = os.path.join(FLAGS.log_dir, meta[-1].split('.')[0])

    saver = tf.train.import_meta_graph(meta_graph)
    sess = tf.Session(config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True))
    with tf.device('/gpu:%s' % FLAGS.gpu):
        saver.restore(sess, meta_weight)
    graph = tf.get_default_graph()
    op = graph.get_tensor_by_name('Generator/decoder/final/tanh:0')
    seg_op = graph.get_tensor_by_name('Softmax_1:0')
    #root = os.path.join(FLAGS.data_dir, FLAGS.fold, 'test')
    #num_subject = np.array(os.listdir(root))
    num_samples = len(test_set)
    pa = FLAGS.output
    if not os.path.isdir(pa):
        os.makedirs(pa)
    avg_psnr = []
    avg_ssim = []
    avg_rmse = []
    DSC_total = []
    for i in range(num_samples):
        input_list = list()
        for num_modalities in modalities[:-1]:
            A_test = os.path.join(data_path, test_set[i], '%s_to_chunktemp_left.nii.gz'%num_modalities)
            img = nib.load(A_test).get_fdata()
            img = img / 100.
            A_test = np.zeros((128, 128, 128))
            A_test[:, :img.shape[1], :img.shape[2]] = img[:128, :, :]
            A_test = np.expand_dims(A_test, axis=0)
            A_test = np.expand_dims(A_test, axis=-1)
            input_list.append(A_test)
        test_input = np.concatenate(input_list, axis=4)
        test_seg_input = 2 * test_input - 1

        test_pet = os.path.join(data_path, test_set[i], '%s_to_chunktemp_left.nii.gz'%modalities[-1])
        test_pet = nib.load(test_pet)
        affine = test_pet.affine
        img_test = test_pet.get_fdata()
        img_test = img_test / 100.
        test_pet = np.zeros((128, 128, 128))
        test_pet[:, :img_test.shape[1], :img_test.shape[2]] = img_test[:128, :, :]
        temp_pet = test_pet * 2 - 1

        test_label = os.path.join(data_path, test_set[i], 'seg-chunktemp-%s-left.nii.gz' % test_set[i])
        test_label = nib.load(test_label)
        label_affine = test_label.affine
        img_label = test_label.get_fdata()
        img = np.asarray(img_label)
        label = np.zeros((128, 128, 128))
        label[:, :img_label.shape[1], :img_label.shape[2]] = img_label[:128, :, :]

        input = graph.get_tensor_by_name('input_holder:0')
        input_3t = graph.get_tensor_by_name('Placeholder_1:0')
        is_train = graph.get_tensor_by_name('is_train_holder:0')
        output1, seg_out = sess.run([op, seg_op], feed_dict={input: test_input, input_3t: test_seg_input, is_train: False})
        seg_out = np.asarray(seg_out)
        seg_out = np.argmax(seg_out, axis=4)
        seg_out = np.squeeze(seg_out)
        dice = dice_eval(seg_out, label)
        print(test_set[i], dice)
        DSC_total.append(dice)

        output = output1[0, :, :, :, 0]
        tem = (output + 1) / 2
        mask = np.ma.masked_where(temp_pet == -1, temp_pet)
        mask = np.ma.getmask(mask)
        masked_temp_pet = temp_pet
        masked_output = copy.deepcopy(output)
        masked_output[mask] = -1
        mask = np.ma.masked_where(temp_pet != -1, temp_pet)
        mask = np.ma.getmask(mask)
        dif = (masked_output - masked_temp_pet) ** 2
        mse_raw = []
        mse2_raw = []
        mse3_raw = []
        for po in range(FLAGS.img_size):
            mse_raw.append(dif[po, :, :][mask[po, :, :]])
            mse2_raw.append(dif[:, po, :][mask[:, po, :]])
            mse3_raw.append(dif[:, :, po][mask[:, :, po]])
        mse = get_mean(mse_raw)
        mse2 = get_mean(mse2_raw)
        mse3 = get_mean(mse3_raw)
        rmse = (np.mean(np.sqrt(mse)) + np.mean(np.sqrt(mse2)) + np.mean(np.sqrt(mse3))) / 3 / 2
        mse = get_psnr_mean(mse_raw, FLAGS.img_size)
        mse2 = get_psnr_mean(mse2_raw, FLAGS.img_size)
        mse3 = get_psnr_mean(mse3_raw, FLAGS.img_size)
        mse[mse == 0] = 1e-10
        mse2[mse2 == 0] = 1e-10
        mse3[mse3 == 0] = 1e-10
        print(test_set[i])
        print('nrmse = %s' % rmse)
        psnr = np.mean(10 * np.log10(4 / mse))
        psnr2 = np.mean(10 * np.log10(4 / mse2))
        psnr3 = np.mean(10 * np.log10(4 / mse3))
        psnr = np.mean([psnr, psnr2, psnr3])
        print('psnr = %s' % psnr)
        ssim = []
        for image in range(output.shape[0]):
            ssim.append(
                compare_ssim(masked_output[image, :, :], masked_temp_pet[image, :, :], data_range=2, win_size=11,
                             gaussian_weights=True))
        ssim = np.mean(ssim)
        print('ssim = %s' % ssim)
        avg_psnr.append(psnr)
        avg_rmse.append(rmse)
        avg_ssim.append(ssim)
        tem[tem < 1e-2] = 0
        dif = test_pet - tem

        img_test[:128, :, :] = tem[:, :img_test.shape[1], :img_test.shape[2]]
        img[:128, :, :] = seg_out[:, :img.shape[1], :img.shape[2]]
        new = nib.Nifti1Image(img_test, affine)
        dif = nib.Nifti1Image(dif, affine)
        seg = nib.Nifti1Image(img, label_affine)
        nib.save(dif, os.path.join(FLAGS.output, 'dif_%s_%s.nii.gz' % (fold, test_set[i])))
        nib.save(new, os.path.join(FLAGS.output, 'synthesis_%s_%s.nii.gz' % (fold, test_set[i])))
        nib.save(seg, os.path.join(FLAGS.output, 'seg_%s_%s.nii.gz' % (fold, test_set[i])))

    psnr = np.mean(avg_psnr)
    ssim = np.mean(avg_ssim)
    rmse = np.mean(avg_rmse)
    print('average nrmse: %s'%rmse)
    print('average psnr: %s'%psnr)
    print('average ssim: %s'%ssim)
    print('dice mean: ', np.mean(np.asarray(DSC_total), axis=0))

if __name__ == "__main__":
    tf.app.run()

import tensorflow as tf
import numpy as np
import os
import ipdb
import nibabel as nib
import scipy.ndimage as sn

data_path1 = 'E:/liuhong/segmentation/codes/SC-GAN-master/data'
subjects = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                '18', '19', '20']
imgs = np.zeros((128, 128, 128))
for subject in subjects:
    data_path = os.path.join(data_path1, subject, 'seg-%s-right.nii.gz' % subject)
    affine = nib.load(data_path).affine
    img = nib.load(data_path).get_fdata()
    #ipdb.set_trace()
    #all
    #img[img > 0] = 1
    #label7
    img[img < 7] = 0
    img[img > 6] = 1
    #label6
    #img[img > 6] = 0
    #img[img < 6] = 0
    #img[img > 5] = 1
    #label5
    #img[img > 5] = 0
    #img[img < 5] = 0
    #img[img > 4] = 1
    #label4
    #img[img > 4] = 0
    #img[img < 4] = 0
    #img[img > 3] = 1
    #label3
    #img[img > 3] = 0
    #img[img < 3] = 0
    #img[img > 2] = 1
    #label2
    #img[img > 2] = 0
    #img[img < 2] = 0
    #img[img > 1] = 1
    #label1
    #img[img > 1] = 0
    #img[img < 1] = 0
    #img[img > 0] = 1

    imgs += img
    #print(img.shape, np.max(img), np.min(img), type(img))
    #mask = nib.Nifti1Image(img, affine)
    #print(img.shape, np.max(img), np.min(img))
    #nib.save(mask, os.path.join(data_path1, subject, 'mask-label-%s-left.nii.gz' % subject))
imgs_ave = imgs/20
#imgs[imgs > 0] = 1
print(np.max(imgs), np.min(imgs))
print(np.max(imgs_ave), np.min(imgs_ave))

data = os.path.join(data_path1, '16', 'seg-16-right.nii.gz')
affine = nib.load(data).affine
seg_ave2 = nib.Nifti1Image(imgs_ave, affine)
nib.save(seg_ave2, os.path.join(data_path1,'train-20-label-mask-right', 'seg_label7_mask.nii.gz'))
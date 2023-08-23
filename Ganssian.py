import tensorflow as tf
import numpy as np
import os
import ipdb
import math
import nibabel as nib

data_path1 = 'E:/liuhong/segmentation/codes/SC-GAN-master/data'
data_path = os.path.join(data_path1, 'train-20-label-mask-right', 'seg_label1_mask.nii.gz')
img = nib.load(data_path).get_fdata()
affine = nib.load(data_path).affine
for i in range(128):
    for j in range(128):
        for k in range(128):
            if img[i][j][k] <= 0.5:
                img[i][j][k] = math.exp(-(math.pow(img[i][j][k] - 0.5, 2) / (2 * (math.pow(0.5, 2)))))
            else:
                img[i][j][k] = math.exp(-(math.pow(img[i][j][k] - 0.5, 2) / (2 * (math.pow(1.0, 2)))))
            # if img[i][j][k] == 0:
            #     img[i][j][k] = math.exp(-(math.pow(img[i][j][k] - 0.5, 2) / (2 * (math.pow(0.5, 2)))))
            # elif 0 < img[i][j][k] < 0.5:
            #     #ipdb.set_trace()
            #     img[i][j][k] = math.exp(-(math.pow(img[i][j][k] - 0.25, 2) / (2 * (math.pow(0.5, 2)))))
            # else:
            #     img[i][j][k] = math.exp(-(math.pow(img[i][j][k] - 0.25, 2) / (2 * (math.pow(1.0, 2)))))


seg_ave2 = nib.Nifti1Image(img, affine)
nib.save(seg_ave2, os.path.join(data_path1,'train-20-label-mask-right', 'seg_label1_mask_G.nii.gz'))


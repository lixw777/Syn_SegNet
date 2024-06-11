#!/bin/bash
#$ -S /bin/bash

set -x -e

for i in {01..20};do
  # Original data
  cd  ./DATASET/ImageNIfTI/${i}

  # 3T and 7T data got from 1st_denoise_n4biascorrection.sh processing
  threett1=3t_t1_s${i}_d_n4.nii.gz
  threett2=3t_t2_s${i}_d_n4.nii.gz
  seventt1=7t_t1_s${i}_d_n4.nii.gz
  seventt2=7t_t2_s${i}_d_n4.nii.gz


  # first subject of 3T and 7T data, used as the template for historgram match
  threett1_01=/home/student9/SC-GAN/data1/brain_data/01/3t_t1_s01_d_n4_trun.nii.gz
  threett2_01=/home/student9/SC-GAN/data1/brain_data/01/3t_t2_s01_d_n4_trun.nii.gz
  seventt1_01=/home/student9/SC-GAN/data1/brain_data/01/7t_t1_s01_d_n4_trun.nii.gz
  seventt2_01=/home/student9/SC-GAN/data1/brain_data/01/7t_t2_s01_d_n4_trun.nii.gz

  ## 3t t1 brain and mask
  for im in $threett1;do
    #truncate  imageIntensity
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz TruncateImageIntensity $im 0.01 0.99
    #rescale image to [0,100]
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz RescaleImage $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz 0 100
    #histogram match to s01
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun_hism.nii.gz HistogramMatch $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz $threett1_01
  done
  for im in  $threett2 ;do
    #truncate  imageIntensity
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz TruncateImageIntensity $im 0.01 0.99
    #rescale image to [0,100]
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz RescaleImage $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz 0 100
    #histogram match to s01
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun_hism.nii.gz HistogramMatch $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz $threett2_01
  done
  for im in  $seventt1 ;do
    #truncate  imageIntensity
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz TruncateImageIntensity $im 0.01 0.99
    #rescale image to [0,100]
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz RescaleImage $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz 0 100
    #histogram match to s01
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun_hism.nii.gz HistogramMatch $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz $seventt1_01
  done
  for im in $seventt2;do
    #truncate  imageIntensity
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz TruncateImageIntensity $im 0.01 0.99
    #rescale image to [0,100]
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz RescaleImage $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz 0 100
    #histogram match to s01
    $ANTSPATH/ImageMath 3 $(basename $im _d_n4.nii.gz)_d_n4_trun_hism.nii.gz HistogramMatch $(basename $im _d_n4.nii.gz)_d_n4_trun.nii.gz $seventt2_01
  done

done

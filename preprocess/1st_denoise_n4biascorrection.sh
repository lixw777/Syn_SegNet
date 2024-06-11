#!/bin/bash
#$ -S /bin/bash

set -x -e
for i in {01..20};do
  # Original 3T and 7T data path(include 20 subjects)
  cd ./DATASET/ImageNIfTI/${i}
  threett1=3t_t1_s${i}.nii.gz
  threett2=3t_t2_s${i}.nii.gz
  seventt1=7t_t1_s${i}.nii.gz
  seventt2=7t_t2_s${i}.nii.gz

  for im in $threett1 $threett2 $seventt1 $seventt2;do
    # Denoise use function from ASHS
    $ASHS_ROOT/ext/Linux/bin/NLMDenoise -i $im -o $(basename $im .nii.gz)_d.nii.gz
    # N4 BiasFieldCorrection in three levels by ANTS
    $ANTSPATH/N4BiasFieldCorrection -d 3 -i $(basename $im .nii.gz)_d.nii.gz -o $(basename $im .nii.gz)_d_n4.nii.gz -s 8 -b [200] -c [50x50x50x50,0.000001]
    $ANTSPATH/N4BiasFieldCorrection -d 3 -i $(basename $im .nii.gz)_d_n4.nii.gz -o $(basename $im .nii.gz)_d_n4.nii.gz -s 4 -b [200] -c [50x50x50x50,0.000001]
    $ANTSPATH/N4BiasFieldCorrection -d 3 -i $(basename $im .nii.gz)_d_n4.nii.gz -o $(basename $im .nii.gz)_d_n4.nii.gz -s 2 -b [200] -c [50x50x50x50,0.000001]
  done
done


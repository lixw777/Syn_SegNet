#!/bin/bash

#$ -S /bin/bash

set -x -e
ATLAS=/home/student9/seg_dataset/ashs_atlas_magdeburg_7t_20180416
# registration template
template=$ATLAS/template/template_0.46.nii.gz
for i in {01..10};do
 for side in {L,R};do
  # Original 3T and 7T data path(include 20 subjects)
  cd  ./DATASET/ImageNIfTI/${i}

  # 3T and 7T data got from 2nd_truncate_rescale_hismatch.sh processing
  threett1=3t_t1_s${i}_d_n4_trun_hism.nii.gz
  threett2=3t_t2_s${i}_d_n4_trun_hism.nii.gz
  seventt1=7t_t1_s${i}_d_n4_trun_hism.nii.gz
  seventt2=7t_t2_s${i}_d_n4_trun_hism.nii.gz

  # hippocampus side(left or right)
  seg=seg-chunktemp-${i}-${side}.nii.gz
  nativeseg=seg-native-${i}-${side}.nii.gz
  threett2chunk=3T2_to_chunktemp_${side}.nii.gz
  threett1chunk=3T1_to_chunktemp_${side}.nii.gz
  seventt2chunk=7t2_to_chunktemp_${side}.nii.gz
  seventt1chunk=7t1_to_chunktemp_${side}.nii.gz
  # space for extracting hippocampus region image
  REFSPACE=$ATLAS/template/refspace_${side}.nii.gz
  threett2_iso=3t_t2_s${i}_d_n4_iso.nii.gz
  seventt2_iso=7t_t2_s${i}_d_n4_iso.nii.gz

  #3T registration
  #t2-->t1
  $ASHS_ROOT/ext/Linux/bin/c3d $threett2 -resample "100x100x200%" -type short -o $threett2_iso # resample to isotropous image
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -a -dof 6 -m MI -n 100x100x10 -i $threett2_iso $threett1 -ia-identity -o 3t_flirt_t2_to_t1.mat # regis t2 to t1
  $ASHS_ROOT/ext/Linux/bin/c3d_affine_tool 3t_flirt_t2_to_t1.mat -inv -o 3t_flirt_t2_to_t1_inv.mat
  #t1-->template
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -a -dof 6 -m NCC 2x2x2 -i $template $threett1 -o 3t_greedy_t1_to_template_init_rigid.mat -n 400x0x0x0 -ia-image-centers -search 400 5 5
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -a -m NCC 2x2x2 -i $template $threett1 -o 3t_greedy_t1_to_template.mat -n 400x80x40x0 -ia 3t_greedy_t1_to_template_init_rigid.mat
  #reslice to template (get whole brain image)
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rm $threett1 3t_reslice_t1_to_template.nii.gz -rf $template -r 3t_greedy_t1_to_template.mat
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rm $threett2 3t_reslice_t2_to_template.nii.gz -rf $template -r 3t_greedy_t1_to_template.mat 3t_flirt_t2_to_t1_inv.mat
  #to template chunk space (get hippocampus region image)
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rf $REFSPACE -rm $threett1 $threett1chunk  -r 3t_greedy_t1_to_template.mat
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rf $REFSPACE -rm $threett2 $threett2chunk  -r 3t_greedy_t1_to_template.mat 3t_flirt_t2_to_t1_inv.mat

  #7T registration
  #t2-->t1
  $ASHS_ROOT/ext/Linux/bin/c3d $seventt2 -resample "100x100x200%" -type short -o $seventt2_iso
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -a -dof 6 -m MI -n 100x100x10 -i $seventt2_iso $seventt1 -ia-identity -o 7t_flirt_t2_to_t1.mat
  $ASHS_ROOT/ext/Linux/bin/c3d_affine_tool 7t_flirt_t2_to_t1.mat -inv -o 7t_flirt_t2_to_t1_inv.mat
  #7t t1--> 3t t1
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -a -dof 6 -m NCC 2x2x2 -i $threett1 $seventt1 -o 7t_greedy_t1_to_3t_init_rigid.mat -n 400x0x0x0 -ia-image-centers -search 400 5 5
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -a -m NCC 2x2x2 -i $threett1 $seventt1 -o 7t_greedy_t1_to_3t.mat -n 400x80x40x0 -ia 7t_greedy_t1_to_3t_init_rigid.mat
  #reslice to template
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rm $seventt1 7t_reslice_t1_to_template.nii.gz -rf $template -r 3t_greedy_t1_to_template.mat 7t_greedy_t1_to_3t.mat
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rm $seventt2 7t_reslice_t2_to_template.nii.gz -rf $template -r 3t_greedy_t1_to_template.mat 7t_greedy_t1_to_3t.mat 7t_flirt_t2_to_t1_inv.mat
  #to template chunk space
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rf $REFSPACE -rm $seventt1 $seventt1chunk  -r 3t_greedy_t1_to_template.mat 7t_greedy_t1_to_3t.mat
  $ASHS_ROOT/ext/Linux/bin/greedy -d 3 -rf $REFSPACE -rm $seventt2 $seventt2chunk -ri LABEL 0.2vox -rm $nativeseg $seg -r 3t_greedy_t1_to_template.mat 7t_greedy_t1_to_3t.mat 7t_flirt_t2_to_t1_inv.mat
  #extract brain
  $FSLPATH/bin/bet 3t_reslice_t1_to_template.nii.gz  3t_reslice_t1_to_template_brain.nii.gz -m -R
  # use brain mask from 3T T1 to extract brain for 3T T2 7T T1 T2
  $FSLPATH/bin/fslmaths 3t_reslice_t1_to_template_brain_mask.nii.gz -mul 3t_reslice_t2_to_template.nii.gz 3t_reslice_t2_to_template_brain.nii.gz
  $FSLPATH/bin/fslmaths 3t_reslice_t1_to_template_brain_mask.nii.gz -mul 7t_reslice_t1_to_template.nii.gz 7t_reslice_t1_to_template_brain.nii.gz
  $FSLPATH/bin/fslmaths 3t_reslice_t1_to_template_brain_mask.nii.gz -mul 7t_reslice_t2_to_template.nii.gz 7t_reslice_t2_to_template_brain.nii.gz
 done
done
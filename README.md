# Syn_SegNet :A Joint Deep Neural Network for Ultrahigh-Field 7T MRI Synthesis and Hippocampal Subfield Segmentation in Routine 3T MRI


This is our implementation of an end-to-end neural network segmentation framework based on the TensorFlow framework, which improves the segmentation accuracy oh hippocampal subfields on routine 3T MRI by synthesizing 7T-like MRI.
![avatar](/imgs/Figure1.jpg)

The paper can be found [here](https://ieeexplore.ieee.org/abstract/document/10218394/algorithms#algorithms) in the IEEE Journal of Biomedical and Health Informatics.



## Prerequisites
- Linux or Windows
- Python 3
- CPU or NVIDIA GPU
- TensorFlow 1.15


## Training
- Train the model
```bash
python training.py 
```

## Testing
- Test the segmentation
```bash
python testing.py
```

##Dataset
This study used a [PAIRED 3T-7T HIPPOCAMPAL SUBFIELD DATASET](https://ieee-dataport.org/documents/paired-3t-7t-hippocampal-subfield-dataset) specially collected for this task (to be uploaded).

## Citation
If you use this code for your research, please cite our paper.

X. Li et al., ["Syn_SegNet: A Joint Deep Neural Network for Ultrahigh-Field 7 T MRI Synthesis and Hippocampal Subfield Segmentation in Routine 3 T MRI."](https://ieeexplore.ieee.org/abstract/document/10218394/algorithms#algorithms) in *IEEE Journal of Biomedical and Health Informatics*, doi: 10.1109/JBHI.2023.3305377.


```
@ARTICLE{10218394,
  author={Li, Xinwei and Wang, Linjin and Liu, Hong and Ma, Baoqiang and Chu, Lei and Dong, Xiaoxi and Zeng, Debin and Che, Tongtong and Jiang, Xiaoming and Wang, Wei and Hu, Jun and Li, Shuyu},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Syn_SegNet: A Joint Deep Neural Network for Ultrahigh-Field 7 T MRI Synthesis and Hippocampal Subfield Segmentation in Routine 3 T MRI}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/JBHI.2023.3305377}}
```




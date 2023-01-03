# CISC-R
Code for TPAMI 2022 paper, [**"Querying Labeled for Unlabeled: Cross-Image Semantic Consistency Guided Semi-Supervised Semantic Segmentation"**](https://ieeexplore.ieee.org/document/10005033/authors#authors).

Authors: Linshan Wu, <a href="https://scholar.google.com/citations?hl=en&user=Gfa4nasAAAAJ">Leyuan Fang</a>, <a href="https://scholar.google.com/citations?hl=zh-CN&user=bHSKDuYAAAAJ">Xingxin He</a>, Min He, <a href="https://scholar.google.com/citations?hl=zh-CN&user=73trMQkAAAAJ">Jiayi Ma</a>, and <a href="https://scholar.google.com/citations?user=nZizkQ0AAAAJ&hl">Zhun Zhong</a>

## Abstract
Semi-supervised semantic segmentation aims to learn a semantic segmentation model via limited labeled images and
adequate unlabeled images. The key to this task is generating reliable pseudo labels for unlabeled images. Existing methods mainly
focus on producing reliable pseudo labels based on the confidence scores of unlabeled images while largely ignoring the use of
labeled images with accurate annotations. In this paper, we propose a Cross-Image Semantic Consistency guided Rectifying (CISC-R)
approach for semi-supervised semantic segmentation, which explicitly leverages the labeled images to rectify the generated pseudo
labels. Our CISC-R is inspired by the fact that images belonging to the same class have a high pixel-level correspondence. Specifically,
given an unlabeled image and its initial pseudo labels, we first query a guiding labeled image that shares the same semantic
information with the unlabeled image. Then, we estimate the pixel-level similarity between the unlabeled image and the queried labeled
image to form a CISC map, which guides us to achieve a reliable pixel-level rectification for the pseudo labels.

## Getting Started
### Prepare Dataset
- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) | [val2017](http://images.cocodataset.org/zips/val2017.zip) | [masks](https://drive.google.com/file/d/166xLerzEEIbU7Mt1UGut-3-VN41FMUb1/view?usp=sharing)
```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
    
├── [Your COCO Path]
    ├── train2017
    ├── val2017
    └── masks
```
### Pretrained Backbone:
[ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth) | [Xception-65](https://drive.google.com/open?id=1_j_mE07tiV24xXOJw4XDze0-a0NAhNVi)
```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    └── xception.pth
```
### Train and Eval
```bash 
python train.py
python eval.py
```
## Acknowledgement
We thank [ST++](https://github.com/LiheYoung/ST-PlusPlus) for part of their codes, processed datasets, data partitions, and pretrained models.

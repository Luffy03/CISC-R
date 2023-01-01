# CISC-R
Code for TPAMI 2023 paper, [**"Querying Labeled for Unlabeled: Cross-Image Semantic Consistency Guided Semi-Supervised Semantic Segmentation"**](https://www.leyuanfang.com/wp-content/uploads/2022/12/TPAMI_CISC_R.pdf).

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

import numpy as np
from PIL import Image
import torch
import os
import cv2
import matplotlib.pyplot as plt
import ttach as tta
from math import *
import torch.nn.functional as F
import torch.nn as nn


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read(img):
    img = Image.open(img)
    return np.asarray(img)


def read_color(img, dataset='pascal'):
    cmap = color_map(dataset)
    img = Image.open(img)
    img = img.convert('P')
    img.putpalette(cmap)
    return img


def write(img, path):
    if type(img) is np.ndarray:
        if len(np.shape(img)) > 2:
            img = img[:, :, ::-1]
        cv2.imwrite(path, img)
    else:
        img.save(path)


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def loss_calc(args, pred, label, reduction='mean'):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    if args.ohem:
        reduce = True if reduction == 'none' else False
        ce = OhemCrossEntropy(ignore_index=255, reduce=reduce)
    else:
        ce = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)

    loss = ce(pred, label)
    return loss


class OhemCrossEntropy(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=100000, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])
        cmap[255] = np.array([255, 255, 255])

    return cmap


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = F.pad(img, (0, 0, rows_missing, cols_missing), 'constant', 0)
    return padded_img


def pre_slide(model, image, num_classes=21, tile_size=(321, 321), tta=False):
    image_size = image.shape  # bigger than (1, 3, 512, 512), i.e. (1,3,1024,1024)
    overlap = 2 / 3  # 每次滑动的重合率为1/2

    stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:769*(1-1/3) = 513
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(1024-769)/513 + 1 = 2
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(2048-769)/513 + 1 = 4

    full_probs = torch.zeros((1, num_classes, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 shape(1024,2048,19)

    count_predictions = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()  # 初始化计数矩阵 shape(1024,2048,19)
    tile_counter = 0  # 滑动计数0

    for row in range(tile_rows):  # row = 0,1
        for col in range(tile_cols):  # col = 0,1,2,3
            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0
            y1 = int(row * stride)  # y1 = 0 * 513 = 0
            x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+769, 2048)
            y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+769, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(769-769, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(769-769, 0)

            img = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:769, 0:769]
            padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为769*769

            tile_counter += 1  # 计数加1
            # print("Predicting tile %i" % tile_counter)

            # 将扣下来的部分传入网络，网络输出概率图。
            # use softmax
            if tta:
                padded = model(padded_img, True)
            else:
                padded = model(padded_img)[0] if isinstance(model(img), tuple) else model(padded_img)
                padded = F.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]  # 扣下相应面积 shape(769,769,19)
            count_predictions[:, :, y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
            full_probs[:, :, y1:y2, x1:x2] += pre  # 窗口区域内的全概率矩阵叠加预测结果

    # average the predictions in the overlapping regions
    full_probs /= count_predictions  # 全概率矩阵 除以 计数矩阵 即得 平均概率

    return full_probs   # 返回整张图的平均概率 shape(1, 1, 1024,2048)



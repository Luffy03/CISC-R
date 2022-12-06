from utils import *
from train import *
from dataset.transform import *
from dataset.semi import SemiDataset
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"


def init_basic_elems(model='deeplabv3plus', backbone='resnet101', dataset='pascal'):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[model](backbone, 21 if dataset == 'pascal' else 19)
    return model.cuda(2)


def random_select(root_path, dataset='pascal', split='1_16'):
    "create original select txt for labeled images"
    check_dir(root_path)
    txt_path = os.path.join(root_path, dataset)
    check_dir(txt_path)
    txt_path = os.path.join(txt_path, split+'/split_0')
    check_dir(txt_path)

    if dataset == 'pascal':
        path = './Pascal'
    else:
        path = './Cityscapes'
    label_txt_path = './splits/%s/%s/split_0/labeled.txt'%(dataset, split)

    with open(label_txt_path, 'r') as f:
        ls = f.read().splitlines()
    print('length of ls:', len(ls))

    for i in tqdm(ls):
        label = read(os.path.join(path, i.split(' ')[1]))
        set = list(np.unique(label))

        # # delete low
        h, w = label.shape
        zeros = np.zeros(label.shape)
        new_set = set.copy()

        for cls in set:
            zeros[label == cls] = 1
            sum = zeros.sum()
            if sum < (h * w * 0.03):
                new_set.remove(cls)
            zeros = np.zeros(label.shape)

        for cls in new_set:
            with open(os.path.join(txt_path, str(cls)+'.txt'), 'a') as f:
                f.write(i + '\n')


def accuracy_select(data_root=None,
                    root_path='./relia_cls_txt', dataset='pascal', split='1_16', model=None):
    check_dir(root_path)
    txt_path = os.path.join(root_path, dataset)
    check_dir(txt_path)
    txt_path = os.path.join(txt_path, split)
    check_dir(txt_path)

    original_txt_path = './cls_txt/%s/%s' % (dataset, split)
    num_classes = 21 if dataset == 'pascal' else 19

    if data_root is None:
        if dataset == 'pascal':
            data_root = './Pascal'
        else:
            data_root = './Cityscapes'

    for index in range(num_classes):
        index_txt_path = original_txt_path + '/' + str(index) + '.txt'
        dataset = SemiDataset(dataset, data_root, 'label', None, None, index_txt_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4,
                                drop_last=False)
        tbar = tqdm(dataloader)

        id_to_reliability = []
        model.eval()
        with torch.no_grad():
            for img, mask, id in tbar:
                img = img.cuda(2)
                mask = mask.cpu().numpy()
                pred = model(img)[0] if isinstance(model(img), tuple) else model(img)
                pred = torch.argmax(pred, dim=1).cpu().numpy()

                metric = meanIOU(num_classes=num_classes)
                metric.add_batch(pred, mask)
                iu, mIOU = metric.evaluate()
                reliability = iu[index]
                tbar.set_description('cls:%d reliability: %.3f' % (index, reliability))
                id_to_reliability.append((id[0], reliability))

        id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
        print(id_to_reliability[len(id_to_reliability) // 2][1])


def select_reliable(dataset='pascal', data_root=None,
                    labeled_id_path=None, unlabeled_id_path=None,
                    pseudo_mask_path=None, model=None,
                    save_labeled_path=None, save_unlabeled_path=None, original_txt_path=None):
    check_dir(save_labeled_path)
    check_dir(save_unlabeled_path)
    num_classes = 21 if dataset == 'pascal' else 19
    crop_size = 321 if dataset == 'pascal' else 721

    if data_root is None:
        if dataset == 'pascal':
            data_root = './Pascal'
        else:
            data_root = './Cityscapes'

    model.eval()
    """
    Get labeled-anchors first, index by num_class
    """
    # mode as train for fast calculation with larger batch_size
    mode = 'train'
    trainset = SemiDataset(dataset, data_root, mode, crop_size,
                           labeled_id_path=labeled_id_path)
    trainset.ids = 4 * trainset.ids if dataset == 'cityscapes' else trainset.ids
    dataloader = DataLoader(trainset, batch_size=8, shuffle=True,
                            pin_memory=True, num_workers=4, drop_last=True)
    tbar = tqdm(dataloader)

    vecs_all = torch.zeros((num_classes, 256)).cuda()
    num = torch.zeros((num_classes)).cuda()
    with torch.no_grad():
        for img, mask, _ in tbar:
            img = img.cuda()
            mask = mask.cuda()
            _, feat = model(img)

            for index in range(num_classes):
                mask_cp = mask.clone()
                mask_cp[mask == index] = 1
                mask_cp[mask != index] = 0
                vec = Weighted_GAP(feat, mask_cp).view(-1)
                vecs_all[index, :] += vec
                num[index] += 1
    # mean
    vecs_all = vecs_all / num

    """
    CISC-based select labeled images for query
    """
    for index in range(num_classes):
        anchor = vecs_all[index]
        index_txt_path = original_txt_path + '/' + str(index) + '.txt'
        labeled_dataset = SemiDataset(dataset, data_root, 'select_labeled', None, labeled_id_path=index_txt_path)
        labeled_dataloader = DataLoader(labeled_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4,
                                drop_last=False)
        tbar = tqdm(labeled_dataloader)

        id_to_reliability = []
        with torch.no_grad():
            for img, mask, id in tbar:
                if index in list(np.unique(mask.cpu().numpy)):
                    break

                img = img.cuda()
                mask = mask.cuda()
                _, feat = model(img)
                n, c, _, _ = feat.size()
                mask_cp = mask.clone()
                mask_cp[mask == index] = 1
                mask_cp[mask != index] = 0

                simi = F.cosine_similarity(feat, anchor.view(n, c, 1, 1))
                simi = F.interpolate(simi.unsqueeze(1), size=mask.size()[-2:], mode='bilinear').view(mask.size())
                bce_loss = F.binary_cross_entropy(simi, mask_cp.float(), reduction='none')
                loss = bce_loss[mask != 255].mean()

                reliability = loss
                # tbar.set_description('reliability: %.3f' % reliability)
                id_to_reliability.append((id[0], reliability))
        id_to_reliability.sort(key=lambda elem: elem[1], reverse=False)
        """
        save labeled images txt for every class
        """
        with open(save_labeled_path + str(index) + '.txt', 'w') as f:
            for elem in id_to_reliability[:len(id_to_reliability) // 2]:
                f.write(elem[0] + '\n')
        print((id_to_reliability[len(id_to_reliability) // 2][1]).cpu().numpy())

    """
    CISC-based select reliable unlabeled images
    """
    unlabeledset = SemiDataset(dataset, data_root, 'select_unlabeled', crop_size,
                           unlabeled_id_path=unlabeled_id_path, pseudo_mask_path=pseudo_mask_path)
    unloader = DataLoader(unlabeledset, batch_size=1, shuffle=True,
                            pin_memory=True, num_workers=4, drop_last=True)
    tbar = tqdm(unloader)
    id_to_reliability = []
    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            mask = mask.cuda()
            mask_np = mask.clone().squeeze(0).cpu().numpy()
            _, feat = model(img)
            n, c, _, _ = feat.size()

            cls_set = list(np.unique(mask_np))
            reliability = 0
            if len(cls_set) == 0:
                reliability = 1e16
            else:
                for cls in cls_set:
                    anchor = vecs_all[cls]
                    mask_cp = mask.clone()
                    mask_cp[mask == cls] = 1
                    mask_cp[mask != cls] = 0
                    simi = F.cosine_similarity(feat, anchor.view(n, c, 1, 1))
                    simi = F.interpolate(simi.unsqueeze(1), size=mask.size()[-2:], mode='bilinear').view(mask.size())
                    bce_loss = F.binary_cross_entropy(simi, mask_cp.float(), reduction='none')
                    loss = bce_loss[mask != 255].mean()
                    reliability += loss

            # tbar.set_description('reliability: %.3f' % reliability)
            id_to_reliability.append((id[0], reliability))
    id_to_reliability.sort(key=lambda elem: elem[1], reverse=False)

    """
    save unlabeled images txt
    """
    with open(save_unlabeled_path+'/reliable_ids.txt', 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(save_unlabeled_path+'/unreliable_ids.txt', 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')

    print((id_to_reliability[len(id_to_reliability) // 2][1]).cpu().numpy())

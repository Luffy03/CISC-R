import numpy as np
import matplotlib.pyplot as plt
from dataset.transform import *
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from utils import *
import torch
from torch.utils.data import DataLoader


class CrossDataset(Dataset):
    def __init__(self, name, root, mode, size,
                 labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None, cls_id_path=None,
                 length=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path
        self.cls_id_path = cls_id_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = self.unlabeled_ids

        else:
            # mode == 'train'
            id_path = labeled_id_path
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if length is not None:
                multiple = length//len(self.ids)
                plus = length - multiple*len(self.ids)
                self.ids = multiple*self.ids + self.ids[:plus]

    def delete_low(self, set, mask, pro=0.03):
        set = list(set)
        if 0 in set and self.name == 'pascal':
            # remove pascal background
            set.remove(0)
        if 255 in set:
            set.remove(255)

        mask = np.asarray(mask)
        h, w = mask.shape
        zeros = np.zeros(mask.shape)
        new_set = set.copy()
        for cls in set:
            zeros[mask == cls] = 1
            sum = zeros.sum()
            if sum < (h*w*pro):
                new_set.remove(cls)
            zeros = np.zeros(mask.shape)
        return new_set

    def get_pair_id(self, set):
        cls_ids_all = []
        for index in set:
            with open(self.cls_id_path + '/' + str(index) + '.txt', 'r') as f:
                cls_ids = f.read().splitlines()
            cls_ids = list(cls_ids)
            cls_ids_all += cls_ids

        if len(cls_ids_all) == 0:
            if self.mode == 'semi_train':
                pair_id = random.sample(self.labeled_ids, 1)[0]
            else:
                pair_id = random.sample(self.ids, 1)[0]
        else:
            if id in cls_ids_all:
                cls_ids_all.remove(id)  # don't repeat
            pair_id = random.sample(cls_ids_all, 1)[0]

        return pair_id

    def get_pair(self, mask, max_num_select=1):
        mask = np.asarray(mask)
        h, w = mask.shape
        set1 = list(np.unique(mask))
        set1 = self.delete_low(set1, mask)

        pair_id = self.get_pair_id(set1)
        pair_img = Image.open(os.path.join(self.root, pair_id.split(' ')[0]))
        pair_mask = Image.open(os.path.join(self.root, pair_id.split(' ')[1]))

        # basic augmentation on all training images
        base_size = 400 if self.name == 'pascal' else 2048
        pair_img, pair_mask = resize(pair_img, pair_mask, base_size, (0.5, 2.0))
        pair_img, pair_mask = hflip(pair_img, pair_mask, p=0.5)

        pair_set = list(np.unique(np.asarray(pair_mask)))
        pair_set = self.delete_low(pair_set, pair_mask)

        # get shared class between labeled and unlabeled images
        select_cls = []
        for k in set1:
            if k in pair_set:
                select_cls.append(k)

        # crop pair mask after selecting class
        # avoid the select cls not exist in pair img caused by cropping
        pair_img_crop, pair_mask_crop = crop(pair_img, pair_mask, self.size)
        if len(select_cls) != 0:
            flag = set(select_cls) & set(np.unique(np.asarray(pair_mask_crop)))
            while not flag:
                # crop again
                pair_img_crop, pair_mask_crop = crop(pair_img, pair_mask, self.size)
                flag = set(select_cls) & set(np.unique(np.asarray(pair_mask_crop)))

        pair_mask_np = np.asarray(pair_mask_crop)

        # get map
        zeros = np.zeros([h, w])
        map1 = zeros.copy()
        map2 = zeros.copy()

        new_pair_set = list(np.unique(np.asarray(pair_mask_crop)))
        # delete again
        new_pair_set = self.delete_low(new_pair_set, pair_mask_crop)

        num_select = 0
        random.shuffle(set1)
        for c in set1:
            if c in new_pair_set:
                num_select += 1
                if num_select > max_num_select:
                    break
                map1[mask == c] = 1
                map2[pair_mask_np == c] = 1

        map1 = torch.from_numpy(map1).long()
        map2 = torch.from_numpy(map2).long()

        return pair_img_crop, pair_mask_crop, map1, map2

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        base_size = 400 if self.name == 'pascal' else 2048
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        # img_str means img with strong augmentation
        if self.mode == 'semi_train':
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)

        # get corresponding pair and map
        pair_img, pair_mask, map1, map2 = self.get_pair(mask)

        img, mask = normalize(img, mask)
        pair_img, pair_mask = normalize(pair_img, pair_mask)

        return img, mask, pair_img, pair_mask, map1, map2, id

    def __len__(self):
        return len(self.ids)

from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import *
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

MODE = None


def parse_args():
    name = 'pascal/1_4'
    # name = 'cityscapes/1_30'
    data = str(name.split('/')[0])
    root = 'Pascal' if data == 'pascal' else 'Cityscapes'

    parser = argparse.ArgumentParser(description='CISC Framework')
    parser.add_argument('--resume_model', type=str,
                        default='./outdir/%s/models/split_0/a.pth'%name)

    # basic settings
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--data-root', type=str, default='./%s' % str(root))
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default=data)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # predict settings
    parser.add_argument('--save-mask-path', type=str, default='outdir/%s/predicts/split_0'%name)

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_mask_path):
        os.makedirs(args.save_mask_path)

    dataset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(dataset, batch_size=1,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)

    model = init_basic_elems(args)
    checkpoint = torch.load(args.resume_model)
    model.load_state_dict(checkpoint)
    print('\nParams: %.1fM' % count_params(model))
    model.eval()

    tbar = tqdm(valloader)
    num_classes = 21 if args.dataset == 'pascal' else 19
    metric = meanIOU(num_classes=num_classes)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img)[0] if isinstance(model(img), tuple) else model(img)
            # pred = pre_slide(model, img, num_classes=num_classes, tile_size=(args.crop_size, args.crop_size))
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]
            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)
            pred.save('%s/%s' % (args.save_mask_path, os.path.basename(id[0].split(' ')[1])))

    mIOU *= 100.0


def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)

    return model.cuda()


if __name__ == '__main__':
    args = parse_args()

    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]

    print()
    print(args)

    main(args)

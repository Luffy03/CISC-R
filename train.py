from dataset.semi import SemiDataset
from dataset.cross import CrossDataset
from dataset.make_query_txt import *
from dataset.transform import *
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import *
import argparse
from copy import deepcopy
import numpy as np
import os
import math
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from apex import amp

os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2"

MODE = None


def parse_args():
    name = 'pascal/1_4'
    # name = 'cityscapes/1_30'
    data = str(name.split('/')[0])
    root = 'Pascal' if data == 'pascal' else 'Cityscapes'

    parser = argparse.ArgumentParser(description='CISC-R Framework')

    # basic settings
    parser.add_argument('--apex', type=bool, default=False)
    parser.add_argument('--name', type=str, default=name)
    parser.add_argument('--data-root', type=str, default='./%s'%str(root))
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default=data)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--ohem', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet101')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default='dataset/splits/%s/split_0/labeled.txt'%name)
    parser.add_argument('--unlabeled-id-path', type=str, default='dataset/splits/%s/split_0/unlabeled.txt'%name)

    parser.add_argument('--pseudo-mask-path', type=str, default='outdir/%s/pseudo_masks/split_0'%name)
    parser.add_argument('--save-path', type=str, default='outdir/%s/models/split_0'%name)

    # arguments for save txt
    parser.add_argument('--reliable-id-path', default='outdir/%s/reliable_id_path/split_0' % name, type=str)

    # arguments for cls_txt_path
    parser.add_argument('--original_cls_path', default='dataset/cls_txt/%s/split_0' % name, type=str)
    args = parser.parse_args()
    return args


def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)
    # apex.amp
    if args.apex:
        model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = DataParallel(model, device_ids=[0, 1, 2])

    else:
        # parallel
        model = DataParallel(model, device_ids=[0, 1, 2])

    return model.cuda(), optimizer


def main(args):
    check_dir(args.save_path)
    check_dir(args.pseudo_mask_path)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/6: Supervised training on labeled images (SupOnly)')

    global MODE
    MODE = 'train'

    trainset = CrossDataset(args.dataset, args.data_root, MODE, args.crop_size,
                            args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path,
                            cls_id_path=args.original_cls_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))
    best_model = train(model, trainloader, valloader, optimizer, args)

    """
        Second stage training CISC with select easy-query unlabeled images
        """
    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 2/6: Pseudo labeling all unlabeled images')
    labelset = SemiDataset(args.dataset, args.data_root, 'label', None, None,
                          unlabeled_id_path=args.unlabeled_id_path)
    dataloader = DataLoader(labelset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 3/6: Select reliable images for the 1st stage re-training')
    # best_model, _ = init_basic_elems(args, load=True)
    select_reliable(data_root=args.data_root,
                    labeled_id_path=args.labeled_id_path, unlabeled_id_path=args.unlabeled_id_path,
                    pseudo_mask_path=args.pseudo_mask_path, dataset=args.dataset, model=best_model,
                    save_labeled_path=args.reliable_id_path + '/CISC_cls_txt/',
                    save_unlabeled_path=args.reliable_id_path,
                    original_txt_path=args.original_cls_path)

    # <================================== The 1st stage re-training ==================================>

    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled '
          'and reliable unlabeled images')
    MODE = 'semi_train'
    selected_cls_id_path = args.reliable_id_path + '/CISC_cls_txt/'
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')

    trainset = CrossDataset(args.dataset, args.data_root, MODE, args.crop_size,
                            args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path,
                            cls_id_path=selected_cls_id_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)
    labelset = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, args.labeled_id_path)
    labelset.ids = labelset.ids * math.ceil(len(trainset.ids) / len(labelset.ids))
    labelloader = DataLoader(labelset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model_teacher = deepcopy(best_model)
    # model_teacher, _ = init_basic_elems(args, load=True)
    model_teacher.eval()
    model, optimizer = init_basic_elems(args)
    best_model = train(model, trainloader, valloader, optimizer, args, model_teacher, labelloader)

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    Labelset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(Labelset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)
    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')
    trainset = CrossDataset(args.dataset, args.data_root, MODE, args.crop_size,
                            args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path,
                            cls_id_path=selected_cls_id_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    labelset = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, args.labeled_id_path)
    labelset.ids = labelset.ids * math.ceil(len(trainset.ids) / len(labelset.ids))
    labelloader = DataLoader(labelset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model_teacher = deepcopy(best_model)
    # model_teacher, _ = init_basic_elems(args, load=True)
    model_teacher.eval()
    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, optimizer, args, model_teacher, labelloader)


def train(model, trainloader, valloader, optimizer, args, model_teacher=None, label_loader=None):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    global MODE

    for epoch in range(args.epochs):
        torch.cuda.empty_cache()
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        loss_m = AverageMeter()
        un_loss_m = AverageMeter()
        seg_loss_m = AverageMeter()
        con_loss_m = AverageMeter()
        tbar = tqdm(trainloader)
        if MODE != 'train':
            label_iter = iter(label_loader)

        for i, inputs in enumerate(tbar):
            if MODE == 'train':
                img1, mask1, img2, mask2, map1, map2, id = inputs
                img1, mask1, img2, mask2, map1, map2 = img1.cuda(), mask1.cuda(), \
                                                       img2.cuda(), mask2.cuda(), \
                                                       map1.cuda(), map2.cuda()
                # # concat
                img = torch.cat([img1, img2], dim=0)
                pred, feat = model(img)
                B = img.size()[0]
                pred1, pred2 = pred[:B//2, :, :, :], pred[B//2:, :, :, :]
                feat1, feat2 = feat[:B//2, :, :, :], feat[B//2:, :, :, :]

                con_loss = consist(MODE, feat1, map1, mask1, feat2, map2, mask2)
                seg_loss = loss_calc(args, pred1, mask1) + loss_calc(args, pred2, mask2)
                loss = seg_loss + con_loss
                con_loss_m.update(con_loss.item(), img1.size()[0])

            else:
                # Mode == 'semi_train'
                # 1 unlabeled, 2 queried labeled
                img1, mask1, img2, mask2, map1, map2, id = inputs
                img1, mask1, img2, mask2, map1, map2 = img1.cuda(), mask1.cuda(), \
                                                           img2.cuda(), mask2.cuda(), \
                                                           map1.cuda(), map2.cuda()
                # jointly-learning with labeled img
                img_l, mask_l, _ = label_iter.next()
                img_l, mask_l = img_l.cuda(), mask_l.cuda()

                # use unlabeled and queried labeled to generate CISC map
                with torch.no_grad():
                    B = img1.size()[0]
                    img = torch.cat([img1, img2], dim=0)
                    _, feat = model_teacher(img)
                    feat1, feat2 = feat[:B, :, :, :], feat[B:, :, :, :]
                    simi_map1 = consist(MODE, feat1, map1, mask1, feat2, map2, mask2)

                # forward labeled and unlabeled
                input_img = torch.cat([img1, img_l], dim=0)
                B = img1.size()[0]
                pred, _ = model(input_img)
                pred_u, pred_l = pred[:B, :, :, :], pred[B:, :, :, :]

                # get loss
                seg_loss = loss_calc(args, pred_l, mask_l)
                un_loss = Weighted_CE(args, pred_u, mask1, map1, simi_map1)
                loss = seg_loss + un_loss
                un_loss_m.update(un_loss.item(), img1.size()[0])

            optimizer.zero_grad()
            if args.apex:
                # use apex.amp to accelerate
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            loss_m.update(loss.item(), img1.size()[0])
            seg_loss_m.update(seg_loss.item(), img1.size()[0])

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            if MODE == 'train':
                tbar.set_description('Loss: %.3f Seg: %.3f Con %.3f' %
                                     (loss_m.avg, seg_loss_m.avg, con_loss_m.avg))
            else:
                tbar.set_description('Loss: %.3f Seg: %.3f un %.3f' %
                                     (loss_m.avg, seg_loss_m.avg, un_loss_m.avg))

        metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)

        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)[0] if isinstance(model(img), tuple) else model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

    return best_model


def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


def Weighted_GAP(supp_feat, mask):
    if len(mask.size()) != 4:
        mask = mask.unsqueeze(1).float()
    if supp_feat.size() != mask.size():
        supp_feat = F.interpolate(supp_feat, size=mask.size()[-2:], mode='bilinear')

    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


def consist(mode, feat1, map1, mask1, feat2, map2, mask2):
    vec2 = Weighted_GAP(feat2, map2)
    simi_map1 = F.cosine_similarity(feat1, vec2)
    simi_map1 = F.interpolate(simi_map1.unsqueeze(1), size=map1.size()[-2:], mode='bilinear')
    simi_map1 = simi_map1.view(map1.size())
    if mode == 'semi_train':
        return simi_map1

    else:
        vec1 = Weighted_GAP(feat1, map1)
        simi_map2 = F.cosine_similarity(feat2, vec1)
        simi_map2 = F.interpolate(simi_map2.unsqueeze(1), size=map2.size()[-2:], mode='bilinear')
        simi_map2 = simi_map2.view(map1.size())

        loss1 = F.binary_cross_entropy(simi_map1, map1.float(), reduction='none')
        loss2 = F.binary_cross_entropy(simi_map2, map2.float(), reduction='none')
        loss = loss1[mask1 != 255].mean() + loss2[mask2 != 255].mean()
        return loss


def Weighted_CE(args, pred, mask, map, simi_map):
    """
            :param pred: the pred for unlabeled imgs
            :param mask: the label of unlabeled imgs
            :param map: the map with shared class-region in unlabeled imgs,
                        built from labeled and unlabeled imgs
            :param simi_map: simi_map on unlabeled imgs
            """
    gap = torch.abs(map - simi_map.detach())
    loss_mat = loss_calc(args, pred, mask, reduction='none')
    loss_mat = (1 - gap) * loss_mat
    return loss_mat.mean()


if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 160}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004}[args.dataset]  # / 8 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 321, 'cityscapes': 721}[args.dataset]

    print()
    print(args)

    main(args)

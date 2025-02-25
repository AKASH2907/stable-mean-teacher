import sys
sys.path.insert(0, '/home/ak119590/activity_detect/caps_net/exp_5_stn_neurips/vidcaps/ucf101/')

import os
import torch
import time
import random
import imageio
import argparse
import numpy as np

import warnings

warnings.filterwarnings("ignore")

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from tqdm import tqdm
from tensorboardX import SummaryWriter

from models.capsules_ucf101 import CapsNet
from models.caps_mvitv2 import CapsNet as CapsNetMViT

from utils.losses import SpreadLoss, DiceLoss
from utils.metrics import get_accuracy, IOU2
from utils.commons import init_seeds


def val_model_interface(minibatch):
    data = minibatch['weak_data'].cuda()
    action = minibatch['action'].cuda()
    label_mask = minibatch['weak_mask'].cuda()
    empty_vector = torch.zeros(action.shape[0]).cuda()

    output, predicted_action, _ = model(data, action)
    class_loss, abs_class_loss = criterion_cls(predicted_action, action)
    loc_loss1 = criterion_loc_1(output, label_mask)
    loc_loss2 = criterion_loc_2(output, label_mask)

    loc_loss = loc_loss1 + loc_loss2
    total_loss = loc_loss + class_loss
    return output, predicted_action, label_mask, action, total_loss, loc_loss, class_loss


def train_model_interface(minibatch):
    data = minibatch['strong_data'].cuda()
    action = minibatch['action'].cuda()
    label_mask = minibatch['strong_mask'].cuda()
    empty_vector = torch.zeros(action.shape[0]).cuda()

    output, predicted_action, _ = model(data, action)

    class_loss, abs_class_loss = criterion_cls(predicted_action, action)
    loc_loss1 = criterion_loc_1(output, label_mask)
    loc_loss2 = criterion_loc_2(output, label_mask)

    loc_loss = loc_loss1 + loc_loss2
    total_loss = loc_loss + class_loss
    return output, predicted_action, label_mask, action, total_loss, loc_loss, class_loss


def train(args, model, train_loader, optimizer, epoch, save_path, writer):
    start_time = time.time()
    steps = len(train_loader)

    model.train(mode=True)
    model.training = True

    total_loss = []
    accuracy = []
    loc_loss = []
    class_loss = []

    start_time = time.time()

    for batch_id, minibatch in enumerate(train_loader):

        optimizer.zero_grad()

        output, predicted_action, _, action, loss, s_loss, c_loss = train_model_interface(minibatch)

        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        loc_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        accuracy.append(get_accuracy(predicted_action, action))

        if (batch_id + 1) % args.pf == 0:
            r_total = np.array(total_loss).mean()
            r_seg = np.array(loc_loss).mean()
            r_class = np.array(class_loss).mean()
            r_acc = np.array(accuracy).mean()

            print(
                f'[TRAIN] epoch-{epoch:0{len(str(args.epochs))}}/{args.epochs}, batch-{batch_id + 1:0{len(str(steps))}}/{steps},' \
                f'loss-{r_total:.3f}, acc-{r_acc:.3f}' \
                f'\t [LOSS ] cls-{r_class:.3f}, seg-{r_seg:.3f}')

            # summary writing
            total_step = (epoch - 1) * len(train_loader) + batch_id + 1
            info_loss = {
                'loss': r_total,
                'loss_seg': r_seg,
                'loss_cls': r_class,
            }
            info_acc = {
                'acc': r_acc
            }

            writer.add_scalars('train/loss', info_loss, total_step)
            writer.add_scalars('train/acc', info_acc, total_step)
            sys.stdout.flush()

        del minibatch, output

    end_time = time.time()
    train_epoch_time = end_time - start_time
    print("Training time: ", train_epoch_time)
    train_total_loss = np.array(total_loss).mean()

    return train_total_loss


def validate(model, val_data_loader, epoch):
    steps = len(val_data_loader)
    model.eval()
    model.training = False
    total_loss = []
    accuracy = []
    loc_loss = []
    class_loss = []
    total_IOU = 0
    validiou = 0
    print('validating...')
    start_time = time.time()

    with torch.no_grad():

        for batch_id, minibatch in enumerate(val_data_loader):

            output, predicted_action, segmentation, action, loss, s_loss, c_loss = val_model_interface(minibatch)
            total_loss.append(loss.item())
            loc_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            accuracy.append(get_accuracy(predicted_action, action))

            maskout = output.cpu()
            maskout_np = maskout.data.numpy()
            # utils.show(maskout_np[0])

            # use threshold to make mask binary
            maskout_np[maskout_np > 0] = 1
            maskout_np[maskout_np < 1] = 0
            # utils.show(maskout_np[0])

            truth_np = segmentation.cpu().data.numpy()
            for a in range(minibatch['weak_data'].shape[0]):
                iou = IOU2(truth_np[a], maskout_np[a])
                if iou == iou:
                    total_IOU += iou
                    validiou += 1

    val_epoch_time = time.time() - start_time
    print("Validation time: ", val_epoch_time)

    r_total = np.array(total_loss).mean()
    r_seg = np.array(loc_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    average_IOU = total_IOU / validiou
    print(f'[VAL] epoch-{epoch}, loss-{r_total:.3f}, acc-{r_acc:.3f} [IOU ] {average_IOU:.3f}')
    sys.stdout.flush()
    return r_total


def parse_args():
    parser = argparse.ArgumentParser(description='add_losses')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--pf', type=int, default=50, help='print frequency every batch')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--loc_loss', type=str, default='dice', help='dice or iou loss')
    parser.add_argument('--exp_id', type=str, default='debug_ucf101', help='experiment name')
    # parser.add_argument('--pkl_file_label', type=str, default="train_annots_8_labeled_random.pkl",
    #                     help='experiment name')
    parser.add_argument('--pkl_file_label', type=str, default="training_annots_with_labels.pkl",
                        help='experiment name')

    parser.add_argument('-at', '--aug_type', type=int, help="0-spatial, 1- temporal, 2 - both, 3-basic(only flip)")

    # define seed params
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    parser.add_argument('--seed_data', type=int, default=47, help='seed variation pickle files')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    init_seeds(args.seed)

    USE_CUDA = True if torch.cuda.is_available() else False
    TRAIN_BATCH_SIZE = args.bs
    VAL_BATCH_SIZE = args.bs
    N_EPOCHS = args.epochs
    LR = args.lr
    loc_loss_criteria = args.loc_loss

    from datasets.sup_ucf_dataloader_st_augs_v1_speedup import UCF101DataLoader, collate_fn_train, collate_fn_test

    labeled_trainset = UCF101DataLoader('train', [224, 224], cl=8, file_id=args.pkl_file_label,
                                        aug_mode=args.aug_type, subset_seed=args.seed_data)
    validationset = UCF101DataLoader('validation', [224, 224], cl=8, file_id="test_annots.pkl",
                                            aug_mode=0, subset_seed=args.seed_data)
    print(len(labeled_trainset), len(validationset))

    labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn_train
    )

    val_data_loader = DataLoader(
        dataset=validationset,
        batch_size=VAL_BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn_test
    )

    print(len(labeled_train_data_loader), len(val_data_loader))

    # # Load pretrained weights
    # model = CapsNet(pretrained_load=True)
    
    model = CapsNetMViT(pretrained_load=True)

    if USE_CUDA:
        model = model.cuda()

    # losses
    global criterion_cls
    global criterion_loc_1
    global criterion_loc_2
    criterion_cls = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion_loc_1 = nn.BCEWithLogitsLoss(size_average=True)
    criterion_loc_2 = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1,
                                                     verbose=True)
    exp_id = args.exp_id
    save_path = os.path.join('sup_train_wts', exp_id)
    model_save_dir = os.path.join(save_path, time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)


    prev_best_val_loss = 10000
    prev_best_train_loss = 10000
    prev_best_val_loss_model_path = None
    prev_best_train_loss_model_path = None

    for e in tqdm(range(1, N_EPOCHS + 1), total=N_EPOCHS, desc="Epochs"):

        train_loss = train(args, model, labeled_train_data_loader, optimizer, e, save_path, writer)

        val_loss = validate(model, val_data_loader, e)

        if train_loss < prev_best_train_loss:
            print("Yay!!! Got the train loss down...")
            train_model_path = os.path.join(model_save_dir, f'best_model_train_loss_{e}.pth')
            torch.save(model.state_dict(), train_model_path)
            prev_best_train_loss = train_loss
            if prev_best_train_loss_model_path:
                os.remove(prev_best_train_loss_model_path)
            prev_best_train_loss_model_path = train_model_path
        scheduler.step(train_loss)

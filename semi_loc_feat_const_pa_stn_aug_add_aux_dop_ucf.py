import sys
sys.path.insert(0, '/home/ak119590/activity_detect/caps_net/exp_5_stn_neurips/vidcaps/ucf101')
import os
import torch
import time
import copy
import random
import argparse
import numpy as np
import os.path as osp

import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.losses import *
from utils import ramps, ramp_ups
from utils.metrics import get_accuracy, IOU2
from utils.helpers import update_ema
from utils.commons import init_seeds, visualize_pred_maps, visualize_rgb_clips


def get_ip_data(data):
    return data['weak_data'].cuda(), data['strong_data'].cuda(), data['weak_mask'].cuda(), data['strong_mask'].cuda(), data['action'].cuda(), data['label'].cuda()


def data_concat(ip1, ip2, dims=0):
    return torch.cat([ip1, ip2], dim=dims)


def val_model_interface(minibatch):
    data = minibatch['weak_data'].cuda()
    action = minibatch['action'].cuda()
    label_mask = minibatch['weak_mask'].cuda()
    empty_vector = torch.zeros(action.shape[0]).cuda()

    st_loc_pred, predicted_action, _ = model(data, action, empty_vector, 0, 0)
    t_loc_pred, predicted_action_ema, _ = ema_model(data, action, empty_vector, 0, 0)

    class_loss, _ = criterion_cls(predicted_action, action)
    loc_loss1 = criterion_loc_1(st_loc_pred, label_mask)
    loc_loss2 = criterion_loc_2(st_loc_pred, label_mask)

    loc_loss_main = loc_loss1 + loc_loss2
    # loc_loss_aux = loc_loss3 + loc_loss4
    loc_loss = loc_loss_main 
    total_loss = loc_loss + class_loss 
    return st_loc_pred, t_loc_pred, predicted_action, label_mask, action, total_loss, class_loss, loc_loss

def train_model_interface(args, label_minibatch, unlabel_minibatch, epoch, global_step, wt_ramp):
    # torch.float32 and weak_label_data.type(torch.cuda.FloatTensor) also equals torch.float32
    weak_label_data, strong_label_data, weak_label_mask, strong_label_mask, label_action, sup_vid_labels = get_ip_data(label_minibatch)
    weak_unlabel_data, strong_unlabel_data, weak_unlabel_mask, strong_unlabel_mask, unlabel_action, unsup_vid_labels = get_ip_data(unlabel_minibatch)
    
    # randomize
    random_indices = torch.randperm(len(sup_vid_labels) + len(unsup_vid_labels))

    # # reshuffle original data
    concat_weak_data = data_concat(weak_label_data, weak_unlabel_data)[random_indices, :, :, :, :]
    concat_strong_data = data_concat(strong_label_data, strong_unlabel_data)[random_indices, :, :, :, :]
    concat_action = data_concat(label_action, unlabel_action)[random_indices]
    concat_weak_loc = data_concat(weak_label_mask, weak_unlabel_mask)[random_indices, :, :, :, :]
    concat_strong_loc = data_concat(strong_label_mask, strong_unlabel_mask)[random_indices, :, :, :, :]
    concat_labels = data_concat(sup_vid_labels, unsup_vid_labels)[random_indices]
    
    # Labeled indexes
    labeled_vid_index = torch.where(concat_labels == 1)[0]

    # STUDENT MODEL
    st_loc_pred, predicted_action_cls, st_action_feat = model(concat_strong_data, concat_action, concat_labels, epoch, args.thresh_epoch)

    # LOC LOSS SUPERVISED - STUDENT
    # labeled predictions
    labeled_st_pred_loc = st_loc_pred[labeled_vid_index]
    # labeled gt
    labeled_gt_loc = concat_strong_loc[labeled_vid_index]
    # calculate losses
    sup_loc_loss_1 = criterion_loc_1(labeled_st_pred_loc, labeled_gt_loc)
    sup_loc_loss_2 = criterion_loc_2(labeled_st_pred_loc, labeled_gt_loc)
    
    if args.aux_4d:
        concat_strong_data_raw_pred = data_concat(concat_strong_data, torch.sigmoid(st_loc_pred.detach()), 1)
        labeled_st_erc_loc_pred = erc_net(concat_strong_data_raw_pred)
        labeled_st_erc_loc_pred = labeled_st_erc_loc_pred[labeled_vid_index]
    else:
        st_erc_loc_pred = erc_net(torch.sigmoid(st_loc_pred.detach()))
        labeled_st_erc_loc_pred = st_erc_loc_pred[labeled_vid_index]
    sup_loc_loss_3 = criterion_loc_1(labeled_st_erc_loc_pred, labeled_gt_loc)
    sup_loc_loss_4 = criterion_loc_2(labeled_st_erc_loc_pred, labeled_gt_loc)
    #############################################################

    # Classification loss SUPERVISED - STUDENT
    class_loss, _ = criterion_cls(predicted_action_cls[labeled_vid_index], concat_action[labeled_vid_index])

    # UPDATE EMA
    update_ema(model, ema_model, global_step, args.ema_val)
    update_ema(erc_net, ema_erc_net, global_step, args.ema_val)

    # TEACHER
    with torch.no_grad():
        t_loc_pred, predicted_action_cls_ema, teacher_action_feat = ema_model(concat_weak_data, concat_action,
                                                                        concat_labels, epoch, args.thresh_epoch)
        # 4D input
        if args.aux_4d:
            concat_weak_data_raw_pred = data_concat(concat_weak_data, torch.sigmoid(t_loc_pred), 1)        
            t_erc_loc_pred = ema_erc_net(concat_weak_data_raw_pred)
        else:
            t_erc_loc_pred = ema_erc_net(torch.sigmoid(t_loc_pred))

    loc_cons_loss_main_orig = loc_const_criterion(st_loc_pred, t_loc_pred)
    loc_cons_loss_main_diff = loc_const_criterion(torch.diff(st_loc_pred, dim=2), torch.diff(t_loc_pred, dim=2))
    
    loc_cons_loss_aux_orig = loc_const_criterion(st_loc_pred, t_erc_loc_pred)
    loc_cons_loss_aux_diff = loc_const_criterion(torch.diff(st_loc_pred, dim=2), torch.diff(t_erc_loc_pred, dim=2))
        
    if args.opt1:
        loc_cons_loss_main = (wt_ramp * loc_cons_loss_main_diff) + ((1 - wt_ramp) * loc_cons_loss_main_orig)
        loc_cons_loss_aux = (wt_ramp * loc_cons_loss_aux_diff) + ((1 - wt_ramp) * loc_cons_loss_aux_orig)
        
    elif args.opt2:
        loc_cons_loss_main = (wt_ramp * loc_cons_loss_main_orig) + ((1 - wt_ramp) * loc_cons_loss_main_diff)
        loc_cons_loss_aux = (wt_ramp * loc_cons_loss_aux_orig) + ((1 - wt_ramp) * loc_cons_loss_aux_diff)

    elif args.opt3:
        loc_cons_loss_main = wt_ramp * (loc_cons_loss_main_orig + loc_cons_loss_main_diff)
        loc_cons_loss_aux = wt_ramp * (loc_cons_loss_aux_orig + loc_cons_loss_aux_diff)
        
    if args.opt4:
        total_cons_loss = loc_cons_loss_aux + loc_cons_loss_main
    elif args.opt5:
        total_cons_loss = (wt_ramp * loc_cons_loss_aux) + loc_cons_loss_main

    sup_loc_loss = sup_loc_loss_1 + sup_loc_loss_2 + sup_loc_loss_3 + sup_loc_loss_4
    total_loss = args.wt_loc * sup_loc_loss + args.wt_cls * class_loss + args.wt_cons * total_cons_loss * wt_ramp

    return st_loc_pred, predicted_action_cls, predicted_action_cls_ema, concat_weak_loc, concat_action, total_loss, sup_loc_loss, class_loss, total_cons_loss, loc_cons_loss_main, loc_cons_loss_aux


def train(args, model, ema_model, erc_net, ema_erc_net, labeled_train_loader, unlabeled_train_loader,
          optimizer, epoch, save_path, writer,
          global_step, ramp_wt):
    start_time = time.time()
    steps = len(unlabeled_train_loader)
    model.train(mode=True)
    model.training = True
    ema_model.train(mode=True)
    ema_model.training = True

    erc_net.train(mode=True)
    ema_erc_net.train(mode=True)

    total_loss = []
    accuracy = []
    acc_ema = []
    sup_loc_loss = []
    class_loss = []
    loc_consistency_loss = []
    loc_cons_main = []
    loc_cons_aux = []

    start_time = time.time()

    labeled_iterloader = iter(labeled_train_loader)

    for batch_id, unlabel_minibatch in enumerate(unlabeled_train_loader):

        global_step += 1

        # u dnt place it between loss.backward and optimizer.step
        # but can place it anywhere else
        optimizer.zero_grad()

        try:
            label_minibatch = next(labeled_iterloader)

        except StopIteration:
            labeled_iterloader = iter(labeled_train_loader)
            label_minibatch = next(labeled_iterloader)

        _, predicted_action, predicted_action_ema, gt_loc_map, action, loss, s_loss, c_loss, cc_loss, lc_loss_main, lc_loss_aux = train_model_interface(
            args, label_minibatch, unlabel_minibatch, epoch, global_step, ramp_wt(epoch))

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        sup_loc_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        loc_consistency_loss.append(cc_loss.item())
        loc_cons_main.append(lc_loss_main.item())
        loc_cons_aux.append(lc_loss_aux.item())
        accuracy.append(get_accuracy(predicted_action, action))
        acc_ema.append(get_accuracy(predicted_action_ema, action))

        if (batch_id + 1) % args.pf == 0:
            r_total = np.array(total_loss).mean()
            # print(r_total)
            r_loc = np.array(sup_loc_loss).mean()
            r_class = np.array(class_loss).mean()
            r_cc_class = np.array(loc_consistency_loss).mean()
            r_lc_main = np.array(loc_cons_main).mean()
            r_lc_aux = np.array(loc_cons_aux).mean()
            r_acc = np.array(accuracy).mean()
            r_acc_ema = np.array(acc_ema).mean()

            print(f'[TRAIN] EPOCH-{epoch:0{len(str(args.epochs))}}/{args.epochs},'
                  f'batch-{batch_id + 1:0{len(str(steps))}}/{steps}'
                  f'\t [LOSS ] loss-{r_total:.3f}, cls-{r_class:.3f}, loc-{r_loc:.3f}, const-{r_cc_class:.3f}, const_main-{r_lc_main:.3f}, const_aux-{r_lc_aux:.3f}'
                  f'\t [ACC] ST-{r_acc:.3f}, T-{r_acc_ema:.3f}')

            # summary writing
            total_step = (epoch - 1) * len(unlabeled_train_loader) + batch_id + 1
            info_loss = {
                'loss': r_total,
                'loss_loc': r_loc,
                'loss_cls': r_class,
                'loss_consistency': r_cc_class,
                'loss_const_main': r_lc_main,
                'loss_const_aux': r_lc_aux
            }
            info_acc = {
                'acc': r_acc,
                'acc_ema': r_acc_ema
            }
            writer.add_scalars('train/loss', info_loss, total_step)
            writer.add_scalars('train/acc', info_acc, total_step)
            sys.stdout.flush()

    end_time = time.time()
    train_epoch_time = end_time - start_time
    print("Training time: ", train_epoch_time)

    train_total_loss = np.array(total_loss).mean()
    return global_step, train_total_loss


def validate(model, erc_net, val_data_loader, epoch):
    steps = len(val_data_loader)
    model.eval()
    model.training = False

    erc_net.eval()

    total_loss = []
    accuracy = []
    acc_ema = []
    sup_loc_loss = []
    class_loss = []
    total_IOU_s = 0
    validiou_s = 0

    total_IOU_t = 0
    validiou_t = 0
    print('\nVALIDATION STARTED...')
    start_time = time.time()

    with torch.no_grad():

        for _, minibatch in enumerate(val_data_loader):
            st_loc_pred, t_loc_pred, predicted_action, gt_loc_map, action, loss, c_loss, s_loss = val_model_interface(minibatch)
            # st_loc_pred, st_loc_pred_aux, predicted_action, predicted_action_ema, gt_loc_map, action, loss, c_loss, s_loss, _, _ = val_model_interface(minibatch)

            # # temporary - ST-Aux evaluation
            # t_loc_pred = st_loc_pred_aux

            total_loss.append(loss.item())
            sup_loc_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            accuracy.append(get_accuracy(predicted_action, action))
            # acc_ema.append(get_accuracy(predicted_action_ema, action))

            # STUDENT
            maskout_s = st_loc_pred.cpu().data.numpy()
            # TEACHER
            maskout_t = t_loc_pred.cpu().data.numpy()
            # utils.show(maskout_s[0])

            # use threshold to make mask binary
            maskout_s[maskout_s > 0] = 1
            maskout_s[maskout_s < 1] = 0

            maskout_t[maskout_t > 0] = 1
            maskout_t[maskout_t < 1] = 0
            # utils.show(maskout_s[0])

            truth_np = gt_loc_map.cpu().data.numpy()
            for a in range(minibatch['weak_data'].shape[0]):
                iou_s = IOU2(truth_np[a], maskout_s[a])
                iou_t = IOU2(truth_np[a], maskout_t[a])
                if iou_s == iou_s:
                    total_IOU_s += iou_s
                    validiou_s += 1

                if iou_t == iou_t:
                    total_IOU_t += iou_t
                    validiou_t += 1

    val_epoch_time = time.time() - start_time
    print("Validation time: ", val_epoch_time)

    r_total = np.array(total_loss).mean()
    r_loc = np.array(sup_loc_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    # r_acc_ema = np.array(acc_ema).mean()
    average_IOU_s = total_IOU_s / validiou_s
    average_IOU_t = total_IOU_t / validiou_t

    # , T-{r_acc_ema:.3f}
    print(f'[VAL] EPOCH-{epoch:0{len(str(args.epochs))}}/{args.epochs}'
          f'\t [LOSS] loss-{r_total:.3f}, cls-{r_class:.3f}, loc-{r_loc:.3f}'
          f'\t [ACC] ST-{r_acc:.3f}' 
          f'\t [IOU ] ST-{average_IOU_s:.3f}, T-{average_IOU_t:.3f}')
    sys.stdout.flush()
    return r_total


if __name__ == '__main__':
    from opts import parse_args
    args = parse_args()
    print(vars(args))
    
    init_seeds(args.seed)

    USE_CUDA = True if torch.cuda.is_available() else False
    if torch.cuda.is_available() and not USE_CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    TRAIN_BATCH_SIZE = args.bs
    VAL_BATCH_SIZE = args.bs
    N_EPOCHS = args.epochs
    LR = args.lr
    
    # LOAD DATASET
    from datasets.ucf_dataloader_st_augs_v1_speedup import UCF101DataLoader, collate_fn_train, collate_fn_test

    labeled_trainset = UCF101DataLoader('train', [224, 224], cl=8, file_id=args.pkl_file_label,
                                        aug_mode=args.aug_type, subset_seed=args.seed_data)
    unlabeled_trainset = UCF101DataLoader('train', [224, 224], cl=8, file_id=args.pkl_file_unlabel,
                                        aug_mode=args.aug_type, subset_seed=args.seed_data)
    validationset = UCF101DataLoader('validation', [224, 224], cl=8, file_id="test_annots.pkl",
                                            aug_mode=0, subset_seed=args.seed_data)

    print(len(labeled_trainset), len(unlabeled_trainset), len(validationset))

    labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=(TRAIN_BATCH_SIZE) // 2,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn_train

    )

    unlabeled_train_data_loader = DataLoader(
        dataset=unlabeled_trainset,
        batch_size=(TRAIN_BATCH_SIZE) // 2,
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

    print(len(labeled_train_data_loader), len(unlabeled_train_data_loader), len(val_data_loader))

    from models.capsules_ucf101_semi_sup_pa import CapsNet
    model = CapsNet()
    
    from models.aux_networks.net_factory_3d import net_factory_3d
    from models.aux_networks.net_factory import net_factory
    # print("UNet 3D")
    # # RGBD - 3D
    if args.aux_nw=='3d':
        print("UNet 3D.....")
        erc_net = net_factory_3d(net_type="unet_3D", in_chns=args.aux_channels, class_num=1)
    elif args.aux_nw=='2d':  
        print("UNet 2D.....")  
        erc_net = net_factory(net_type="unet_2D", in_chns=args.aux_channels, class_num=1)
    elif args.aux_nw=='2d3d':
        print("UNet 2D3D.....")
        erc_net = net_factory_3d(net_type="unet_2D3D", in_chns=1, class_num=1)
    else:
        print("No aux networks selected!!!!!!")

    # Load pretrained weights
    if args.burn_in:
        model.load_previous_weights(osp.join(args.burn_wts))
        erc_net.load_state_dict(torch.load(osp.join(args.burn_aux_wts)), strict=True)

    if USE_CUDA:
        model = model.cuda()
        erc_net = erc_net.cuda()

    ema_model = copy.deepcopy(model)
    ema_erc_net = copy.deepcopy(erc_net)
    
    if args.opt4:
        print("Run-name - Both main+aux loss added same weight w/ any rampup...")
    elif args.opt5:
        print("Run-name - Aux loss ramp up till ramp thresh epochs and then same wt both main+aux...")

    if args.opt1:
        print("Ramp up - DoP, Ramp down - L2, based on ramp thresh epochs...")
    elif args.opt2:
        print("Ramp up - L2, Ramp down - DoP, based on ramp thresh epochs...")
    elif args.opt3:
        print("Ramp up DoP + L2 both, based on ramp thresh epochs...")
        
    # losses
    global criterion_cls
    global criterion_loc_1
    global criterion_loc_2
    global loc_const_criterion
    global_step = 0

    criterion_cls = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion_loc_1 = nn.BCEWithLogitsLoss(size_average=True)
    criterion_loc_2 = DiceLoss()

    if args.const_loss == 'jsd':
        loc_const_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    elif args.const_loss == 'l2':
        loc_const_criterion = nn.MSELoss()

    elif args.const_loss == 'l1':
        loc_const_criterion = nn.L1Loss()
    
    elif args.const_loss == 'dice':
        loc_const_criterion = DiceLoss()

    print("Loc consistency criterion: ", loc_const_criterion)

    optimizer = optim.Adam(list(model.parameters()) + list(erc_net.parameters()), lr=LR, weight_decay=0,
                           eps=1e-6)
    
    if args.scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1,
                                                        verbose=True)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 45], gamma=0.1, verbose=True)

    ramp_wt = ramp_ups.sigmoid_rampup(args.ramp_thresh)

    exp_id = args.exp_id
    save_path = osp.join('./train_log_wts', exp_id)
    model_save_dir = osp.join(save_path, time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not osp.exists(model_save_dir):
        os.makedirs(model_save_dir)

    prev_best_train_loss = 10000
    prev_best_train_loss_model_path_aux = None
    prev_best_train_loss_model_path_main = None

    gs = 0

    for e in tqdm(range(1, N_EPOCHS + 1), total=N_EPOCHS, desc="Epochs"):
        gs, train_loss = train(args, model, ema_model, erc_net, ema_erc_net, labeled_train_data_loader,
                               unlabeled_train_data_loader,
                               optimizer, e, save_path, writer, global_step, ramp_wt)
        global_step = gs
        val_loss = validate(model, erc_net, val_data_loader, e)

        if train_loss < prev_best_train_loss:
            print("Yay!!! Got the train loss down...")
            # paths
            train_model_path = osp.join(model_save_dir, f'best_model_train_loss_{e}.pth')
            train_model_path_aux = osp.join(model_save_dir, f'best_aux_model_train_loss_{e}.pth')
            
            # save weights only
            torch.save(model.state_dict(), train_model_path)
            prev_best_train_loss = train_loss
            if prev_best_train_loss_model_path_main and e<25:
                os.remove(prev_best_train_loss_model_path_main)
            prev_best_train_loss_model_path_main = train_model_path

            torch.save(erc_net.state_dict(), train_model_path_aux)
            if prev_best_train_loss_model_path_aux and e<25:
                os.remove(prev_best_train_loss_model_path_aux)
            prev_best_train_loss_model_path_aux = train_model_path_aux
        if args.thresh_epoch<e<=args.thresh_epoch+1:
            print(prev_best_train_loss)
        
            prev_best_train_loss +=5
            print(prev_best_train_loss)
            
        if args.scheduler:
            scheduler.step(train_loss)

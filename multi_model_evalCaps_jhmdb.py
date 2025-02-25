import os
import csv
import glob
import torch
import argparse
import numpy as np
import os.path as osp

from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import DataLoader
from utils.commons import init_seeds
from models.capsules_jhmdb_semi_sup_pa import CapsNet


def iou():
    """
    Calculates the accuracy, f-mAP, and v-mAP over the test set
    """
    from datasets.jhmdb_dataloader_eval import JHMDB21Dataset
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--ckpt', type=str, help='experiment name')
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    args = parser.parse_args()
    
    init_seeds(args.seed)

    # declare csvs
    eval_scores = osp.join(args.ckpt, "scores")
    Path(eval_scores).mkdir(parents=True, exist_ok=True)
    v_map_thresh = open(osp.join(eval_scores, "v_map_thresh.csv"), "w+")
    f_map_thresh = open(osp.join(eval_scores, "f_map_thresh.csv"), "w+")
    v_map_class = open(osp.join(eval_scores, "v_map_classes.csv"), "w+")
    f_map_class = open(osp.join(eval_scores, "f_map_classes.csv"), "w+")

    iou_threshs = np.arange(0, 20, dtype=np.float32)/20
    num_classes = np.arange(21)

    csv.writer(v_map_thresh).writerow(["epoch_num"] + [str(i) for i in iou_threshs])
    csv.writer(f_map_thresh).writerow(["epoch_num"] + [str(i) for i in iou_threshs])

    csv.writer(v_map_class).writerow(["epoch_num"] + [str(i) for i in num_classes])
    csv.writer(f_map_class).writerow(["epoch_num"] + [str(i) for i in num_classes])

    model = CapsNet().cuda()
    
    n_classes = 21
    clip_batch_size = 14

    model_names = list()
    fmap_best = list()
    vmap_best = list()

    filtered_files = [file.split("/")[-1] for file in glob.glob(osp.join(args.ckpt, 'best_model_train_' + '*.pth'))]

    for saved_wts in sorted(glob.glob(osp.join(args.ckpt, 'best_model_train_' + '*.pth'))):
        model.load_previous_weights(saved_wts)
        model_names.append(saved_wts)
        model.eval()
        model.training = False

        with torch.no_grad():

            validationset = JHMDB21Dataset('test',[224, 224], 8)
            val_data_loader = DataLoader(
                dataset=validationset,
                batch_size=1,
                num_workers=8,
                shuffle=False
            )
            
            n_correct, n_vids, n_tot_frames = 0, np.zeros((n_classes, 1)), np.zeros((n_classes, 1))

            # 20 IoUs List
            frame_ious = np.zeros((n_classes, 20))
            video_ious = np.zeros((n_classes, 20))
            iou_threshs = np.arange(0, 20, dtype=np.float32)/20

            for sample in tqdm(val_data_loader, total=len(val_data_loader), desc="testing..."):
                video, bbox, label = sample
                video = video[0]
                bbox = bbox[0]
                label = label[0]

                f_skip = 2
                clips = []
                n_frames = video.shape[0]
                # print(n_frames)
                for i in range(0, video.shape[0], 8*f_skip):
                    for j in range(f_skip):
                        b_vid, b_bbox = [], []
                        for k in range(8):
                            ind = i + j + k*f_skip
                            # print(ind)
                            if ind >= n_frames:
                                b_vid.append(np.zeros((1, 224, 224, 3), dtype=np.float32))
                                b_bbox.append(np.zeros((1, 224, 224, 1), dtype=np.float32))
                            else:
                                b_vid.append(video[ind:ind+1, :, :, :])
                                b_bbox.append(bbox[ind:ind+1, :, :, :])
                        clips.append((np.concatenate(b_vid, axis=0), np.concatenate(b_bbox, axis=0), label))
                        if np.sum(clips[-1][1]) == 0:
                            clips.pop(-1)

                if len(clips) == 0:
                    print('Video has no bounding boxes')
                    continue

                batches, gt_segmentations = [], []
                # print(len(clip_batch_size))
                for i in range(0, len(clips), clip_batch_size):
                    x_batch, bb_batch, y_batch = [], [], []
                    for j in range(i, min(i+clip_batch_size, len(clips))):
                        x, bb, y = clips[j]
                        x_batch.append(x)
                        bb_batch.append(bb)
                        y_batch.append(y)
                    batches.append((x_batch, bb_batch, y_batch))
                    gt_segmentations.append(np.stack(bb_batch))

                gt_segmentations = np.concatenate(gt_segmentations, axis=0)
                gt_segmentations = gt_segmentations.reshape((-1, 224, 224, 1))  # Shape N_FRAMES, 112, 112, 1

                segmentations, predictions = [], []
                frames = []
                for x_batch, bb_batch, y_batch in batches:
                    data = np.transpose(np.array(x_batch), [0, 4, 1, 2, 3])
                    data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
                    empty_action = np.ones((len(x_batch),1),np.int)*500
                    empty_action = torch.from_numpy(empty_action).cuda()
                    # print(empty_action)

                    segmentation, pred, _ = model(data, empty_action, empty_action, 0, 0)
                    segmentation = torch.sigmoid(segmentation)
                    segmentation_np = segmentation.cpu().data.numpy()   # B x C x F x H x W -> B x 1 x 8 x 224 x 224
                    segmentation_np = np.transpose(segmentation_np, [0, 2, 3, 4, 1])   
                    save_clip = data.cpu().data.numpy()
                    save_clip = np.transpose(save_clip, [0, 2, 3, 4, 1])
                    # print(segmentation_np.shape, data.shape, save_clip.shape) 
                    frames.append(save_clip)
                    segmentations.append(segmentation_np)
                    predictions.append(pred.cpu().data.numpy())

                # print(f'preds length: {len(predictions)}')
                predictions = np.concatenate(predictions, axis=0)
                #predictions = predictions.reshape((-1, n_classes))
                assert predictions.shape[1] == n_classes
                fin_pred = np.mean(predictions, axis=0)

                fin_pred = np.argmax(fin_pred)
                if fin_pred == label:
                    n_correct += 1

                pred_segmentations = np.concatenate(segmentations, axis=0)
                pred_segmentations = pred_segmentations.reshape((-1, 224, 224, 1))

                pred_segmentations = (pred_segmentations >= 0.5).astype(np.int64)
                seg_plus_gt = pred_segmentations + gt_segmentations

                frames_save = np.concatenate(frames, axis=0)
                frames_save = frames_save.reshape((-1, 224, 224, 3))

                vid_inter, vid_union = 0, 0
                # calculates f_map
                for i in range(gt_segmentations.shape[0]):
                    frame_gt = gt_segmentations[i]
                    if np.sum(frame_gt) == 0:
                        continue

                    n_tot_frames[label] += 1

                    inter = np.count_nonzero(seg_plus_gt[i] == 2)
                    union = np.count_nonzero(seg_plus_gt[i])
                    vid_inter += inter
                    vid_union += union

                    i_over_u = inter / union
                    for k in range(iou_threshs.shape[0]):
                        if i_over_u >= iou_threshs[k]:
                            frame_ious[label, k] += 1

                n_vids[label] += 1
                i_over_u = vid_inter / vid_union
                for k in range(iou_threshs.shape[0]):
                    if i_over_u >= iou_threshs[k]:
                        video_ious[label, k] += 1
                # if np.sum(n_vids)==100: break

            # print('Accuracy:', n_correct / np.sum(n_vids))

            fAP = frame_ious/n_tot_frames
            fmAP = np.mean(fAP, axis=0)
            vAP = video_ious/n_vids
            vmAP = np.mean(vAP, axis=0)

            print(f'Accuracy: {n_correct / np.sum(n_vids):.3f}, IoU/fmap/vmap: {iou_threshs[10]}: {fmAP[10]:.3f} {vmAP[10]:.3f},\
                fmap/vmap@[0.5:0.95]: {np.mean(fmAP[10:]):.3f} {np.mean(vmAP[10:]):.3f}')

            csv.writer(v_map_thresh).writerow(["epoch_" + saved_wts.split('.')[0].split("_")[-1]] + ['{:.3f}'.format(i) for i in vmAP])
            csv.writer(f_map_thresh).writerow(["epoch_" + saved_wts.split('.')[0].split("_")[-1]] + ['{:.3f}'.format(i) for i in fmAP])

            csv.writer(v_map_class).writerow(["epoch_" + saved_wts.split('.')[0].split("_")[-1]] + ['{:.3f}'.format(i) for i in vAP[:, 10]])
            csv.writer(f_map_class).writerow(["epoch_" + saved_wts.split('.')[0].split("_")[-1]] + ['{:.3f}'.format(i) for i in fAP[:, 10]])
            
            fmap_best.append(fmAP[10])
            vmap_best.append(vmAP[10])

    best_fmap_model = model_names[fmap_best.index(max(fmap_best))]
    best_vmap_model = model_names[vmap_best.index(max(vmap_best))]
    best_files = list()
    best_files.append(best_fmap_model)
    best_files.append(best_vmap_model)
    # print(filtered_files)
    # exit()
    # aux_models = glob.glob(osp.join(args.ckpt, "*aux*"))
    
    for file in filtered_files:
        if osp.join(args.ckpt, file) not in best_files:
            os.remove(osp.join(args.ckpt, file))
            try:
                os.remove(osp.join(args.ckpt, "best_aux_model_train_loss_" + file.split(".")[0].split("_")[-1] + ".pth"))
            except:
                continue
    # for aux_file in aux_models:
    #     if aux_file.split("/")[-1].split(".")[0].split("_")[-1] != 
    print(os.listdir(args.ckpt))


iou()


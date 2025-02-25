import os
import cv2
import h5py
import pickle
import os.path as osp
import numpy as np
from pathlib import Path
from scipy.io import loadmat


_dataset_dir = '/home/c3-0/datasets/JHMDB/videos'
_mask_dir = '/home/c3-0/datasets/JHMDB_puppet_mask/puppet_mask'
dest_path = "/home/ak119590/datasets/JHMDB21_h5"

def load_video(video_name):
    video_dir = os.path.join(_dataset_dir, '%s.avi' % video_name)
    # try:
    cap = cv2.VideoCapture(video_dir)
    video = []
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        video.append(frame)

    video = np.array(video)
    mask_dir = os.path.join(_mask_dir, '{}/puppet_mask.mat'.format(video_name))
    mat_data = loadmat(mask_dir)
    mask_m = mat_data['part_mask']
    mask = np.zeros((mask_m.shape[2], 256, 256))
    for m in range(mask_m.shape[2]):
        mask[m] = cv2.resize(mask_m[:, :, m], (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, -1)
    annot_frames = np.arange(mask.shape[0])
    # print(video.shape, mask.shape, annot_frames)
    # print(annot_frames)
    # exit()        


    # except:
    #     print('Error:', str(video_dir))
    #     print('Error:', str(mask_dir))
    #     return None, None, None, None
    # save info
    save_path = osp.join(dest_path, video_name)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    with h5py.File(osp.join(save_path, 'clip_info.h5'), 'w') as hf:
        hf.create_dataset('rgb', data=video)
        hf.create_dataset('loc_map', data=mask)
        hf.create_dataset('annots', data=annot_frames)
        hf.flush()
    # exit()


if __name__ == '__main__':
    from tqdm import tqdm

    # training_annot_file = "/home/ak119590/activity_detect/caps_net/exp_5_stn_neurips/vidcaps/jhmdb21_data_lists/jhmdb_seed_37/jhmdb_classes_list_per_70_unlabeled.txt"
    training_annot_file = "/home/ak119590/activity_detect/caps_net/exp_5_stn_neurips/vidcaps/jhmdb21_data_lists/testlist.txt"

    train_list = open(training_annot_file, "r").readlines()

    for i in tqdm(range(len(train_list)), total=len(train_list)):
        video_name = train_list[i].rstrip()
        load_video(video_name)
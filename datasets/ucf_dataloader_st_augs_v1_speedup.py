import os
import numpy as np
import random
from skvideo.io import vread
import torch
import h5py
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import cv2
from torchvision import transforms
from .spatial_aug import *
from .temporal_aug import *


class UCF101DataLoader(Dataset):
    def __init__(self, mode, clip_shape, cl, file_id, aug_mode, subset_seed):
        self._dataset_dir = 'UCF101_dataset_path' 
        # self._dataset_dir = '/data/akash/UCF101_24/'
        # self._dataset_dir = '/data/akash/UCF101_24/'
        self.subset_seed = subset_seed
        
        if mode == 'train':
            self.vid_files = self.get_det_annots_prepared(file_id)
            self.shuffle = True
            self.mode = 'train'

        else:
            self.vid_files = self.get_det_annots_test_prepared(file_id)
            self.shuffle = False
            self.mode = 'test'

        self._height = clip_shape[0]
        self._width = clip_shape[1]
        self.cl = cl
        self._size = len(self.vid_files)
        self.indexes = np.arange(self._size)

        self.toPIL = transforms.ToPILImage()
        self.erase_size = 25
        self.aug_mode = aug_mode
        

    def get_det_annots_prepared(self, file_id):
        import pickle

        training_annot_file = "../data_lists/data_subset_pkl_files_seed_{}/".format(str(self.subset_seed)) + file_id

        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)

        return training_annotations

    def get_det_annots_test_prepared(self, file_id):
        import pickle
        testing_anns = "../data_lists/test_annots.pkl"
        with open(testing_anns, 'rb') as ts_rid:
            testing_annotations = pickle.load(ts_rid)

        return testing_annotations

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):

        v_name, anns = self.vid_files[index]

        clip, bbox_clip, label, annot_frames, labeled_vid = self.load_video(v_name, anns)
        depth = self.cl

        if clip is None:
            if self.mode == 'train':
                return None, None, None, None, None, None, None
            else:
                return None, None, None, None

        vlen, clip_h, clip_w, _ = clip.shape
        vskip = 2

        if len(annot_frames) == 1:
            selected_annot_frame = annot_frames[0]
        else:
            if len(annot_frames) <= 0:
                if self.mode == 'train':
                    return None, None, None, None, None, None, None
                else:
                    return None, None, None, None

            # Get the annotation index
            try:
                annot_idx = np.random.randint(0, len(annot_frames) - int((depth * vskip)))
            except:
                try:
                    annot_idx = np.random.randint(0, len(annot_frames) - int((depth * vskip) / 2))
                except:
                    if self.mode == 'train':
                        return None, None, None, None, None, None, None
                    else:
                        return None, None, None, None

            # choose frame_idx from annotations
            selected_annot_frame = annot_frames[annot_idx]

        # Get the start frame
        start_frame = selected_annot_frame
        # - int((depth * vskip)/2)

        if start_frame < 0:
            vskip = 1
            start_frame = selected_annot_frame
            # - int((depth * vskip)/2)
            if start_frame < 0:
                start_frame = 0
                vskip = 1
        if selected_annot_frame >= vlen:
            if self.mode == 'train':
                return None, None, None, None, None, None, None
            else:
                return None, None, None, None

        #  Now here we will cover all frames with anns or 
        # most of them with second try option
        if start_frame + (depth * vskip) >= vlen:
            start_frame = vlen - (depth * vskip)

        # frame index to chose - 0, 2, 4, ..., 16
        span = (np.arange(depth) * vskip)

        # APPLY TEMPORAL AUG
        if self.aug_mode==1 or self.aug_mode == 2:
            random_span = np.array(sorted(np.random.choice(max(span), self.cl, replace=False)))
            video, bbox_clip = get_temp_aug_view(clip, bbox_clip, start_frame, span, random_span)
        else:
            # frame_ids
            span += start_frame
            # print(span)
            video = clip[span]
            bbox_clip = bbox_clip[span]

        # 8, 240, 320, 3
        weak_aug_video = list()
        strong_aug_video = list()
        weak_aug_bbox = list()
        strong_aug_bbox = list()

        # # take random crops for training
        # crop_area = np.random.uniform(0.6, 1)
        # print(crop_area)

        if self.mode == 'train':
            start_pos_h = np.random.randint(0, clip_h - self._height)
            start_pos_w = np.random.randint(0, clip_w - self._width)
            # for frame in range(video.shape[0]):
            # simple_frm, strong_frm, simple_bbox_aug, strong_bbox_aug = self.train_augs(video[frame], bbox_clip[frame], clip_h, clip_w)
            # weak_aug_video[frame], strong_aug_video[frame], weak_aug_bbox[frame], strong_aug_bbox[frame] 
            
            # APPLY SPATIAL AUGS
            if self.aug_mode != 3:
                weak_aug_video, strong_aug_video, weak_aug_bbox, strong_aug_bbox, aug_probab_array = \
                    get_aug_views(video, bbox_clip, self._height, self._width, start_pos_h, start_pos_w, self.mode, self.erase_size) 
            else:
                weak_aug_video, strong_aug_video, weak_aug_bbox, strong_aug_bbox, aug_probab_array = \
                    get_basic_aug_views(video, bbox_clip, self._height, self._width, start_pos_h, start_pos_w, self.mode, self.erase_size) 
            if self.aug_mode==1:
                strong_aug_video = weak_aug_video
                strong_aug_bbox = weak_aug_bbox

            weak_aug_video = torch.stack(weak_aug_video)
            strong_aug_video = torch.stack(strong_aug_video)
            weak_aug_bbox = torch.stack(weak_aug_bbox)
            strong_aug_bbox = torch.stack(strong_aug_bbox)
            weak_aug_video = weak_aug_video.permute(1, 0, 2, 3)
            strong_aug_video = strong_aug_video.permute(1, 0, 2, 3)
            weak_aug_bbox = weak_aug_bbox.permute(1, 0, 2, 3)
            strong_aug_bbox = strong_aug_bbox.permute(1, 0, 2, 3)

        else:
            # center crop for validation
            start_pos_h = int((clip_h - self._height) / 2)
            start_pos_w = int((clip_w - self._width) / 2)
            weak_aug_video, weak_aug_bbox, aug_probab_array = \
                get_aug_views(video, bbox_clip, self._height, self._width, start_pos_h, start_pos_w, self.mode, self.erase_size)

            weak_aug_video = torch.stack(weak_aug_video)
            weak_aug_bbox = torch.stack(weak_aug_bbox)
            weak_aug_video = weak_aug_video.permute(1, 0, 2, 3)
            weak_aug_bbox = weak_aug_bbox.permute(1, 0, 2, 3)

        action_tensor = torch.Tensor([label])
        labeled_vid = torch.Tensor([labeled_vid])
        aug_probab_array = torch.Tensor(aug_probab_array)

        if self.mode == 'train':
            # sample = {'weak_data':weak_aug_video, 'strong_data': strong_aug_video, 'weak_mask':weak_aug_bbox, 'strong_mask':strong_aug_bbox, 'action':action_tensor, 'label':labeled_vid, 'aug_probab': aug_probab_array}
            return weak_aug_video, strong_aug_video, weak_aug_bbox, strong_aug_bbox, action_tensor, labeled_vid, aug_probab_array
        else:

            # sample = {'weak_data':weak_aug_video, 'strong_data': weak_aug_video, 'weak_mask':weak_aug_bbox, 'strong_mask':weak_aug_bbox, 'action':action_tensor, 'label':labeled_vid, 'aug_probab':aug_probab_array}
            return weak_aug_video, weak_aug_bbox, action_tensor, labeled_vid

    def load_video(self, video_name, annotations):
        
        try:
            src_path = '/home/ak119590/datasets/UCF101_24_h5'
            vid_info = h5py.File(osp.join(src_path, video_name, 'clip_info.h5'), "r")
            video = np.array(vid_info['rgb'])
            bbox = np.array(vid_info['loc_map'])
            multi_frame_annot = list(vid_info['annots'])
            label, labeled_vid = annotations[0][2], annotations[0][5]
        except:
            print("ERROR:", video_name)
            return None, None, None, None, None
        
        if self.mode == 'train':
            return video, bbox, label, multi_frame_annot, labeled_vid
        else:
            return video, bbox, label, multi_frame_annot, labeled_vid


def collate_fn_train(batch):
    weak_aug_video, strong_aug_video, weak_aug_bbox, strong_aug_bbox, action_tensor, \
    labeled_vid, aug_probab_array = [], [], [], [], [], [], []

    for item in batch:
        if not (None in item):
            weak_aug_video.append(item[0])
            strong_aug_video.append(item[1])
            weak_aug_bbox.append(item[2])
            strong_aug_bbox.append(item[3])
            action_tensor.append(item[4])
            labeled_vid.append(item[5])
            aug_probab_array.append(item[6])

    weak_aug_video = torch.stack(weak_aug_video)
    strong_aug_video = torch.stack(strong_aug_video)
    weak_aug_bbox = torch.stack(weak_aug_bbox)
    strong_aug_bbox = torch.stack(strong_aug_bbox)
    action_tensor = torch.stack(action_tensor)
    labeled_vid = torch.stack(labeled_vid)
    aug_probab_array = torch.stack(aug_probab_array)

    sample = {'weak_data': weak_aug_video, 'strong_data': strong_aug_video, 'weak_mask': weak_aug_bbox,
              'strong_mask': strong_aug_bbox, 'action': action_tensor, 'label': labeled_vid,
              'aug_probab': aug_probab_array}
    return sample


def collate_fn_test(batch):
    weak_aug_video, weak_aug_bbox, action_tensor, labeled_vid = [], [], [], []

    for item in batch:
        if not (None in item):
            weak_aug_video.append(item[0])
            weak_aug_bbox.append(item[1])
            action_tensor.append(item[2])
            labeled_vid.append(item[3])

    weak_aug_video = torch.stack(weak_aug_video)
    weak_aug_bbox = torch.stack(weak_aug_bbox)
    action_tensor = torch.stack(action_tensor)
    labeled_vid = torch.stack(labeled_vid)

    sample = {'weak_data': weak_aug_video, 'weak_mask': weak_aug_bbox, 'action': action_tensor, 'label': labeled_vid}
    return sample


def pil_to_cv(pil_image):
    """
    Returns a copy of an image in a representation suited for OpenCV
    :param pil_image: PIL.Image object
    :return: Numpy array compatible with OpenCV
    """
    return np.array(pil_image)[:, :, ::-1]


def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h = 224, 224
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(pil_to_cv(frame))
        # print(frame.shape)
        # writer.write(frame)

    writer.release()


if __name__ == '__main__':
    from pathlib import Path
    import random
    import time
    from tqdm import tqdm

    def init_seeds(seed=2563, cuda_deterministic=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
        if cuda_deterministic:  # slower, more reproducible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:  # faster, less reproducible
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    test_mode = "train"

    if test_mode=="train":    
        dataset = UCF101DataLoader('train', [224, 224], 16, file_id='train_annots_90_unlabeled_random.pkl', aug_mode=0, subset_seed=47)

        verify_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn_train)
        save_path = 'dataloader_viz/verify_new_aug_loader/train_time/'
    else:
        dataset = UCF101DataLoader('test', [224, 224], 32, file_id='train_annots_10_labeled_random.pkl')

        verify_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_test)
        save_path = 'dataloader_viz/verify_new_aug_loader/test_time/'
        
    print(len(verify_dataloader))
    # exit()
    
    Path(save_path).mkdir(parents=True, exist_ok=True)

    vid_vis_counter = 0

    start = time.time()
    for i, data in tqdm(enumerate(verify_dataloader), total=len(verify_dataloader)):
        if (i+1) % 25 == 0:
            print("Data iter:", i)
        orig_clip  = data['weak_data']
        clip_mask = data['weak_mask']

        if test_mode=="train":
            aug_clip = data['strong_data']
            strong_mask = data['strong_mask']
            aug_probab_array = data['aug_probab']

        vid_class = data['action']
        vid_labeled = data['label']
        

        if orig_clip.shape[0] != 32:
            print(orig_clip.shape, aug_clip.shape)
            print(clip_mask.shape, strong_mask.shape)

        # if i==25:
        #     break

    print(time.time() - start)
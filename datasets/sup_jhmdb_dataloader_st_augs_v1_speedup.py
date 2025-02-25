import os
import numpy as np
import random
from skvideo.io import vread
import torch
import h5py
import os.path as osp
from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
from torchvision import transforms
from .spatial_aug import *
from .temporal_aug import *

'''
Instead of video clips frame selection we are making sure 
to select frames from annotation length. That way annotation
will not be empty for any case.

Apply temporally consistent strong spatial augmentaions:
 - ColorJitter - Hue, Brightness, Saturation, Contrast
 - Gaussian Blur
 - Grayscale

'''


class JHMDB21DataLoader(Dataset):
    def __init__(self, mode, clip_shape, cl, file_id,  aug_mode, subset_seed):
        # self._dataset_dir = '/home/akash/dataset/JHMDB'
        # self._mask_dir = '/home/akash/dataset/puppet_mask'
        self._dataset_dir = 'home/c3-0/datasets/JHMDB/videos'
        self._mask_dir = '/home/c3-0/datasets/JHMDB_puppet_mask/puppet_mask'

        self.subset_seed = subset_seed

        self._class_list = ['brush_hair', 'catch', 'clap', 'climb_stairs',
                            'golf', 'jump', 'kick_ball', 'pick', 'pour',
                            'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow',
                            'shoot_gun', 'sit', 'stand', 'swing_baseball',
                            'throw', 'walk', 'wave']

        if mode == 'train':
            self.vid_files = self.get_det_annots_prepared(file_id)
            self.shuffle = True
            self.mode = 'train'
        else:
            self.vid_files = self.get_det_annots_test_prepared()
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

        training_annot_file = "../jhmdb21_data_lists/jhmdb_seed_{}/".format(str(self.subset_seed)) + file_id

        with open(training_annot_file, "r") as rid:
            train_list = rid.readlines()

        for i in range(len(train_list)):
            train_list[i] = train_list[i].rstrip()

        print("Training samples from :", training_annot_file)

        return train_list

    def get_det_annots_test_prepared(self):

        test_annot_file = '../jhmdb21_data_lists/testlist.txt'

        with open(test_annot_file, "r") as rid:
            test_file_list = rid.readlines()

        for i in range(len(test_file_list)):
            test_file_list[i] = test_file_list[i].rstrip()

        print("Testing samples from :", test_annot_file)

        return test_file_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):

        v_name = self.vid_files[index]

        clip, bbox_clip, label, annot_frames = self.load_video(v_name)
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

        # take random crops for training
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
                weak_aug_video.append(w1)
                strong_aug_video.append(s1)
                weak_aug_bbox.append(wb1)
                strong_aug_bbox.append(sb1)

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
        # labeled_vid = torch.Tensor([labeled_vid])
        aug_probab_array = torch.Tensor(aug_probab_array)

        if self.mode == 'train':
            # sample = {'weak_data':weak_aug_video, 'strong_data': strong_aug_video, 'weak_mask':weak_aug_bbox, 'strong_mask':strong_aug_bbox, 'action':action_tensor, 'label':labeled_vid, 'aug_probab': aug_probab_array}
            return weak_aug_video, strong_aug_video, weak_aug_bbox, strong_aug_bbox, action_tensor, aug_probab_array
        else:

            # sample = {'weak_data':weak_aug_video, 'strong_data': weak_aug_video, 'weak_mask':weak_aug_bbox, 'strong_mask':weak_aug_bbox, 'action':action_tensor, 'label':labeled_vid, 'aug_probab':aug_probab_array}
            return weak_aug_video, weak_aug_bbox, action_tensor
        # return sample


    def load_video(self, video_name):
        try:
            src_path = "/home/ak119590/datasets/JHMDB21_h5"
            vid_info = h5py.File(osp.join(src_path, video_name, 'clip_info.h5'), "r")
            video = np.array(vid_info['rgb'])
            mask = np.array(vid_info['loc_map'])
            annot_frames = list(vid_info['annots'])
            label = self._class_list.index(video_name.split('/')[0])
        except:
            print('Error:', str(video_dir))
            print('Error:', str(mask_dir))
            return None, None, None, None

        return video, mask, label, annot_frames


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
            # labeled_vid.append(item[5])
            aug_probab_array.append(item[5])

    weak_aug_video = torch.stack(weak_aug_video)
    strong_aug_video = torch.stack(strong_aug_video)
    weak_aug_bbox = torch.stack(weak_aug_bbox)
    strong_aug_bbox = torch.stack(strong_aug_bbox)
    action_tensor = torch.stack(action_tensor)
    # labeled_vid = torch.stack(labeled_vid)
    aug_probab_array = torch.stack(aug_probab_array)

    sample = {'weak_data': weak_aug_video, 'strong_data': strong_aug_video, 'weak_mask': weak_aug_bbox,
              'strong_mask': strong_aug_bbox, 'action': action_tensor, 'aug_probab': aug_probab_array}
    return sample


def collate_fn_test(batch):
    weak_aug_video, weak_aug_bbox, action_tensor, labeled_vid = [], [], [], []

    for item in batch:
        if not (None in item):
            weak_aug_video.append(item[0])
            weak_aug_bbox.append(item[1])
            action_tensor.append(item[2])
            # labeled_vid.append(item[3])

    weak_aug_video = torch.stack(weak_aug_video)
    weak_aug_bbox = torch.stack(weak_aug_bbox)
    action_tensor = torch.stack(action_tensor)
    # labeled_vid = torch.stack(labeled_vid)

    sample = {'weak_data': weak_aug_video, 'weak_mask': weak_aug_bbox, 'action': action_tensor}
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
    import imageio
    import skvideo.io
    from skvideo.io import ffprobe
    from pathlib import Path
    import random

    seed = 295145
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = JHMDB('train', [224, 224], 8, "jhmdb_classes_list_per_70_unlabeled.txt")
    print(len(dataset))
    # exit()
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_train)
    print(len(train_dataloader))

    save_path = 'dataloader_viz/jhmdb_load_data_debug/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    vid_vis_counter = 0

    for i, data in enumerate(train_dataloader):
        # if i%25==0:
        print("Data iter:", i)
        orig_clip, aug_clip = data['weak_data'], data['strong_data']
        clip_mask = data['weak_mask']
        strong_mask = data['strong_mask']

        vid_class = data['action']
        # vid_labeled = data['label']
        aug_probab_array = data['aug_probab']

        # check collate function
        # if orig_clip.shape[0]!=8:
        #     print(orig_clip.shape, aug_clip.shape)
        #     print(clip_mask.shape, strong_mask.shape)
        #     print(vid_class, aug_probab_array)
        #     exit()

        # AUG-VIZ
        orig_clip = np.transpose(orig_clip.numpy(), [0, 2, 3, 4, 1])
        aug_clip = np.transpose(aug_clip.numpy(), [0, 2, 3, 4, 1])
        clip_mask = np.transpose(clip_mask.numpy(), [0, 2, 3, 4, 1])
        strong_mask = np.transpose(strong_mask.numpy(), [0, 2, 3, 4, 1])

        for v in range(1):

            # CLIP ANALYSIS
            filename = os.path.join(save_path, 'orig_clip_{}.mp4'.format(vid_vis_counter))
            write_video(filename, (orig_clip[v] * 255).astype(np.uint8), 25)

            filename = os.path.join(save_path, 'aug_clip_{}.mp4'.format(vid_vis_counter))
            write_video(filename, (aug_clip[v] * 255).astype(np.uint8), 25)

            filename = os.path.join(save_path, 'mask_clip_{}.mp4'.format(vid_vis_counter))

            final_msk_vid = list()
            for x in range(orig_clip[v].shape[0]):
                im = (orig_clip[v, x, :, :, :] * 255).astype(np.uint8)
                msk = (clip_mask[v, x, :, :, 0] * 255).astype(np.uint8)
                final_im = cv2.bitwise_and(im, im, mask=msk)
                final_msk_vid.append(final_im)
            final_msk_vid = np.array(final_msk_vid)
            # final_msk_vid = np.expand_dims(final_msk_vid, axis=3)
            # print(final_msk_vid.shape)
            write_video(filename, final_msk_vid, 25)

            # # FRAME BY FRAME ANALYSIS
            # with imageio.get_writer('{}/orig_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
            #     for x in range(orig_clip[v].shape[0]):
            #         image = (orig_clip[v,x,:,:,:]*255).astype(np.uint8)
            #         image = image[...,::-1].copy()
            #         writer.append_data(image) 

            # with imageio.get_writer('{}/aug_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
            #     for x in range(aug_clip[v].shape[0]):
            #         image = (aug_clip[v,x,:,:,:]*255).astype(np.uint8)
            #         image = image[...,::-1].copy()
            #         writer.append_data(image) 

            # with imageio.get_writer('{}/mask_orig_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
            #     for x in range(orig_clip[v].shape[0]):
            #         image = (orig_clip[v,x,:,:,:]*255).astype(np.uint8)
            #         cl_mask = (clip_mask[v,x,:,:,0]*255).astype(np.uint8)
            #         image = cv2.bitwise_and(image, image, mask=cl_mask)
            #         image = image[...,::-1].copy()
            #         writer.append_data(image) 
            # with imageio.get_writer('{}/mask_aug_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
            #     for x in range(aug_clip[v].shape[0]):
            #         image = (aug_clip[v,x,:,:,:]*255).astype(np.uint8)
            #         cl_mask = (strong_mask[v,x,:,:,0]*255).astype(np.uint8)
            #         image = cv2.bitwise_and(image, image, mask=cl_mask)
            #         image = image[...,::-1].copy()
            #         writer.append_data(image) 

            vid_vis_counter += 1
        if i > 5: break
        # exit()

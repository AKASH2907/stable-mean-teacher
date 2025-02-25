import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import cv2
import h5py
import os
from skvideo.io import vread


class UCF101Dataset(Dataset):
    def __init__(self, mode, clip_shape, cl, file_id=None):
        self._dataset_dir = '/home/ke005409/Datasets/UCF101'  # CRCV cluster
        # self._dataset_dir = '/data/akash/UCF101_24/'
        # self._dataset_dir = '/data/akash/UCF101_24/'

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
            

    def get_det_annots_prepared(self, file_id):
        import pickle

        training_annot_file = "../data_lists/data_subset_pkl_files_seed_{}/".format(str(self.subset_seed)) + file_id

        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)

        return training_annotations
        
        
    def get_det_annots_test_prepared(self):
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
        clip, bbox_clip, label = self.load_video(v_name, anns)
        
        # Center crop
        _, clip_h, clip_w, _ = clip.shape

        start_pos_h = int((clip_h - self._height) / 2)
        start_pos_w = int((clip_w - self._width) / 2)
        
        clip = clip[:, start_pos_h:start_pos_h+self._height, start_pos_w:start_pos_w+self._width, :] / 255.
        bbox_clip = bbox_clip[:, start_pos_h:start_pos_h+self._height, start_pos_w:start_pos_w+self._width, :]
        return clip, bbox_clip, label
    

    def load_video(self, video_name, annotations):
        
        try:
            src_path = '/home/ak119590/datasets/UCF101_24_h5'
            vid_info = h5py.File(osp.join(src_path, video_name, 'clip_info.h5'), "r")
            video = np.array(vid_info['rgb'])
            bbox = np.array(vid_info['loc_map'])
            label = annotations[0][2]
        except:
            print("ERROR:", video_name)
            return None, None, None, None, None
        
        if self.mode == 'train':
            return video, bbox, label
        else:
            return video, bbox, label
        


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
    from pathlib import Path
    from torch.utils.data import DataLoader
    mode='test'
    clip_shape=[224,224]
    channels=3
    batch_size = 1
    dataset = UCF101Dataset(mode, clip_shape, batch_size, False)
    print(len(dataset))
    

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    print(len(train_dataloader))
    # exit()
    save_path = 'dataloader_viz/resize_erase_crop_debug/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    vid_vis_counter = 0

    for i, data in enumerate(train_dataloader):
        if i%25==0:
            print("Data iter:", i)
        orig_clip, aug_clip = data['weak_data'], data['strong_data']
        clip_mask = data['weak_mask']
        strong_mask = data['strong_mask']

        vid_class = data['action']
        vid_labeled = data['label']
        aug_probab_array = data['aug_probab']

        if orig_clip.shape[0]!=8:
            print(orig_clip.shape, aug_clip.shape)
            print(clip_mask.shape, strong_mask.shape)
        # print(vid_class, vid_labeled, aug_probab_array)
        # exit()
        # check collate function
        # if orig_clip.shape[0]!=8:
        #     print(orig_clip.shape, aug_clip.shape)
        #     print(clip_mask.shape, strong_mask.shape)
        #     print(data['label'])
        #     exit()

        # print(vid_class, vid_labeled)
        
        # # AUG-VIZ
        orig_clip = np.transpose(orig_clip.numpy(), [0, 2, 3, 4, 1])
        aug_clip = np.transpose(aug_clip.numpy(), [0, 2, 3, 4, 1])
        clip_mask = np.transpose(clip_mask.numpy(), [0, 2, 3, 4, 1])
        strong_mask = np.transpose(strong_mask.numpy(), [0, 2, 3, 4, 1])

        # # if i>10: break
        # # exit()

        # for v in range(8):

            # # CLIP ANALYSIS
            # filename = os.path.join(save_path, 'orig_clip_{}.mp4'.format(vid_vis_counter))
            # write_video(filename, (orig_clip[v]*255).astype(np.uint8), 25)

            # filename = os.path.join(save_path, 'aug_clip_{}.mp4'.format(vid_vis_counter))
            # write_video(filename, (aug_clip[v]*255).astype(np.uint8), 25)

            # filename = os.path.join(save_path, 'mask_clip_{}.mp4'.format(vid_vis_counter))
            

            # final_msk_vid = list()
            # for x in range(orig_clip[v].shape[0]):
            #     im  = (orig_clip[v,x,:,:,:]*255).astype(np.uint8)
            #     msk = (clip_mask[v,x,:,:,0]*255).astype(np.uint8)
            #     final_im = cv2.bitwise_and(im, im, mask=msk)
            #     final_msk_vid.append(final_im)
            # final_msk_vid = np.array(final_msk_vid)
            # # final_msk_vid = np.expand_dims(final_msk_vid, axis=3)
            # # print(final_msk_vid.shape)
            # write_video(filename, final_msk_vid, 25)
            
        #     # FRAME BY FRAME ANALYSIS
        #     with imageio.get_writer('{}/orig_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
        #         for x in range(orig_clip[v].shape[0]):
        #             image = (orig_clip[v,x,:,:,:]*255).astype(np.uint8)
        #             writer.append_data(image) 

        #     with imageio.get_writer('{}/aug_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
        #         for x in range(aug_clip[v].shape[0]):
        #             image = (aug_clip[v,x,:,:,:]*255).astype(np.uint8)
        #             writer.append_data(image) 

        #     with imageio.get_writer('{}/mask_orig_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
        #         for x in range(orig_clip[v].shape[0]):
        #             image = (orig_clip[v,x,:,:,:]*255).astype(np.uint8)
        #             cl_mask = (clip_mask[v,x,:,:,0]*255).astype(np.uint8)
        #             image = cv2.bitwise_and(image, image, mask=cl_mask)
        #             writer.append_data(image) 
        #     with imageio.get_writer('{}/mask_aug_clip_{:02d}.gif'.format(save_path, vid_vis_counter), mode='I') as writer:
        #         for x in range(aug_clip[v].shape[0]):
        #             image = (aug_clip[v,x,:,:,:]*255).astype(np.uint8)
        #             cl_mask = (strong_mask[v,x,:,:,0]*255).astype(np.uint8)
        #             image = cv2.bitwise_and(image, image, mask=cl_mask)
        #             writer.append_data(image) 
        #     vid_vis_counter+=1
        # # if i>50: break
        # exit()
      
      
      
      





'''
#exit()
    index = 0
    while True:
        # [clip, lbl_mask, cls_mask], [lbl, cls_lbl] = dataloader.__getitem__(index)
        clip, flip_clip, lbl_mask, cls_lbl = dataloader.__getitem__(index)
        print(clip.shape, lbl_mask.shape, cls_lbl)
        exit()
        # with imageio.get_writer('./test_visual/orig_{:02d}_gt.gif'.format(index), mode='I') as writer:
        #     for i in range(0, clip.shape[0], 20):
        #         image = (clip[i]*255).astype(np.uint8)
        #         writer.append_data(image) 
        # with imageio.get_writer('./test_visual/flip_{:02d}_gt.gif'.format(index), mode='I') as writer:
        #     for i in range(0, flip_clip.shape[0], 20):
        #         image = (flip_clip[i]*255).astype(np.uint8)
        #         writer.append_data(image) 
        
        #print(frm_idx)
        #pdb.set_trace()
        #with imageio.get_writer('./results/{:02d}_gt.gif'.format(index), mode='I') as writer:
        #    for i in range(clip.shape[1]):
        #        image = (clip[0,i]*255).astype(np.uint8)
        #        #image = image[:,:,::-1]
        #        writer.append_data(image) 
      
        # for i in range(cls_lbl.shape[1]):
        #     for j in range(cls_lbl.shape[-1]):
        #         img = cls_lbl[0, i, :, :, j] * 255
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)
                
        #         img = cls_mask[0, i, :, :, j] * 255
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_mask_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)
            
        #     img = lbl[0, i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)            
            
        #     img = lbl_mask[0, i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_fg_mask_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)                        
                
        #out_img = './results/{:02d}_fg.jpg'.format(index)
        #img = lbl[0,0,:,:,0] * 255
        #cv2.imwrite(out_img, img)
      
      
        with imageio.get_writer('./results/{:02d}_lbl.gif'.format(index), mode='I') as writer:
            for i in range(clip.shape[1]):
              image = (lbl[0,i] * clip[0,i]).astype(np.uint8) 
              writer.append_data(image) 


        for j in range(mask.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_mcls.gif'.format(index, j), mode='I') as writer:
          for i in range(mask.shape[1]):
            image = (mask[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    
        
        for j in range(mask_mul.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_mmul.gif'.format(index, j), mode='I') as writer:
          for i in range(mask_mul.shape[1]):
            image = (mask_mul[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    

        for j in range(mask_add.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_madd.gif'.format(index, j), mode='I') as writer:
          for i in range(mask_add.shape[1]):
            image = (mask_add[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    
'''


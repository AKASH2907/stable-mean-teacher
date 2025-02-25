import os
import h5py
import pickle
import os.path as osp
import numpy as np
from skvideo.io import vread
from pathlib import Path

dataset_dir = '/home/ke005409/Datasets/UCF101'
dest_path = "/home/ak119590/datasets/UCF101_24_h5"

def load_video(video_name, annotations):
    video_dir = os.path.join(dataset_dir, 'UCF101_Videos/%s.avi' % video_name)
    # print(video_dir)
    try:
        video = vread(str(video_dir))  # Reads in the video into shape (F, H, W, 3)
        # print(video.shape)
    except:
        print('Error:', str(video_dir))
        return None, None, None, None, None

    # creates the bounding box annotation at each frame
    n_frames, h, w, ch = video.shape
    # n_frames
    bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)

    multi_frame_annot = []  # annotations[annot_idx][4]
    bbox_annot = np.zeros((n_frames, h, w, 1), dtype=np.uint8)

    for ann in annotations:
        # fill array with frame ids where action is happening
        multi_frame_annot.extend(ann[4])

        # get strt, end, video class and whether its labeled/unlabled set
        start_frame, end_frame, label, labeled_vid = ann[0], ann[1], ann[2], ann[5]
        # if len(annotations) > 1:
        #     print(start_frame, end_frame)
        collect_annots = []
        # loop over min of video shape or # of frames in annotations
        for f in range(start_frame, min(n_frames, end_frame + 1)):
            try:
                x, y, w, h = ann[3][f - start_frame]
                bbox[f, y:y + h, x:x + w, :] = 1
                # append if frame id present in annotation
                if f in ann[4]:
                    collect_annots.append([x, y, w, h])
            except:
                print('ERROR LOADING ANNOTATIONS')
                print(start_frame, end_frame)
                print(video_dir)
                exit()
        # Expect to have collect_annots with same length as annots for this set 
        # [ c, c, c, c, c, c ....]
        # it's already sorted but sort it for safety
        select_annots = ann[4]
        select_annots.sort()

        # if no frame id in annotations
        if len(collect_annots) == 0:
            continue

        # x_min, y_min, width, height
        [x, y, w, h] = collect_annots[0]
        if len(collect_annots) == 1:
            bbox_annot[start_frame:end_frame, y:y + h, x:x + w, :] = 1
        else:
            bbox_annot[start_frame:select_annots[0], y:y + h, x:x + w, :] = 1
            for i in range(len(collect_annots) - 1):
                frame_diff = select_annots[i + 1] - select_annots[i]
                if frame_diff > 1:
                    [x, y, w, h] = collect_annots[i]
                    pt1 = np.array([x, y, x + w, y + h])
                    [x, y, w, h] = collect_annots[i + 1]
                    pt2 = np.array([x, y, x + w, y + h])
                    points = np.linspace(pt1, pt2, frame_diff).astype(np.int32)
                    for j in range(points.shape[0]):
                        [x1, y1, x2, y2] = points[j]
                        bbox_annot[select_annots[i] + j, y1:y2, x1:x2, :] = 1
                else:
                    [x, y, w, h] = collect_annots[i]
                    bbox_annot[select_annots[i], y:y + h, x:x + w, :] = 1
            [x, y, w, h] = collect_annots[-1]
            bbox_annot[select_annots[-1]:end_frame, y:y + h, x:x + w, :] = 1

    # multiple overlaps when there are multiple annotations for a video so called set
    multi_frame_annot = list(set(multi_frame_annot))
    # print(label, labeled_vid)
    # exit()
    # # save info
    save_path = osp.join(dest_path, video_name)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    with h5py.File(osp.join(save_path, 'clip_info.h5'), 'w') as hf:
        hf.create_dataset('rgb', data=video)
        hf.create_dataset('loc_map', data=bbox)
        hf.create_dataset('annots', data=multi_frame_annot)
        hf.flush()

    # exit()
    # if self.mode == 'train':
    #     # return video, bbox_annot, label, multi_frame_annot, labeled_vid, len(annotations)
    # return video, bbox, label, multi_frame_annot, labeled_vid
    # else:
    #     return video, bbox, label, multi_frame_annot, labeled_vid


if __name__ == '__main__':
    from tqdm import tqdm
    training_annot_file = "../../data_lists/data_subset_pkl_files_seed_47/train_annots_10_labeled_random.pkl"

    with open(training_annot_file, 'rb') as tr_rid:
        training_annotations = pickle.load(tr_rid)


    # print(len(training_annot_file))
    for i in tqdm(range(len(training_annotations)), total=len(training_annotations)):
        video_name = training_annotations[i][0]
        annotation = training_annotations[i][1]
        load_video(video_name, annotation)
    

    training_annot_file = "../../data_lists/data_subset_pkl_files_seed_47/train_annots_90_unlabeled_random.pkl"

    with open(training_annot_file, 'rb') as tr_rid:
        training_annotations = pickle.load(tr_rid)


    # print(len(training_annot_file))
    for i in tqdm(range(len(training_annotations)), total=len(training_annotations)):
        video_name = training_annotations[i][0]
        annotation = training_annotations[i][1]
        load_video(video_name, annotation)

    training_annot_file = "../../data_lists/test_annots.pkl"

    with open(training_annot_file, 'rb') as tr_rid:
        training_annotations = pickle.load(tr_rid)


    # print(len(training_annot_file))
    for i in tqdm(range(len(training_annotations)), total=len(training_annotations)):
        video_name = training_annotations[i][0]
        annotation = training_annotations[i][1]
        load_video(video_name, annotation)
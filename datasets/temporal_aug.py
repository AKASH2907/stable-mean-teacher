import numpy as np

def get_temp_aug_view(clip, bbox_clip, start_frame, span, random_span):
    ta = np.random.choice(3)
    # print(ta) 
    # NO Aug
    if ta == 0:
        span += start_frame
        video = clip[span]
        bbox_clip = bbox_clip[span]
    # random frames in sequential order
    elif ta == 1:
        random_span += start_frame
        video = clip[random_span]
        bbox_clip = bbox_clip[random_span]
    # temporal reverse
    elif ta == 2:
        span += start_frame
        span = np.flip(span)
        video = clip[span]
        bbox_clip = bbox_clip[span]
    
    return video, bbox_clip
    
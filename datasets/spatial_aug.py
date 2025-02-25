import cv2
import numpy as np
from torchvision import transforms


def get_aug_views(video, bbox_clip, clip_h, clip_w, start_pos_h,
                                                   start_pos_w, modes, erase_size):
    aug_probab_array = np.random.rand(7)

    cropping_factor = np.random.uniform(0.6, 1)  # on an average cropping factor is 80% i.e. covers 64% area
    x0 = np.random.randint(0, 224 - 224 * cropping_factor + 1)
    y0 = np.random.randint(0, 224 - 224 * cropping_factor + 1)

    x_erase = np.random.randint(0, 224 - erase_size)
    y_erase = np.random.randint(0, 224 - erase_size)
    erase_size_w = np.random.randint(int(erase_size / 2), erase_size)
    erase_size_h = np.random.randint(int(erase_size / 2), erase_size)

    contrast_val = np.random.uniform(0.6, 1.4)
    hue_val = np.random.uniform(-0.1, 0.1)
    saturation_val = np.random.uniform(0.6, 1.4)
    brightness_val = np.random.uniform(0.6, 1.4)
    gamma_val = np.random.uniform(0.6, 1.4)

    # print(aug_probab_array, contrast_val,
    #             hue_val, saturation_val, brightness_val, gamma_val, cropping_factor, x0, y0, erase_size_w,
    #             erase_size_h, x_erase, y_erase)
    # exit()

    weak_aug_video = list()
    strong_aug_video = list()
    weak_aug_bbox = list()
    strong_aug_bbox = list()
    if modes=='train':
        for frame in range(video.shape[0]):
            w1, s1, wb1, sb1 = train_augs(video[frame], bbox_clip[frame], clip_h, clip_w, start_pos_h, start_pos_w, aug_probab_array, contrast_val,
                hue_val, saturation_val, brightness_val, gamma_val, cropping_factor, x0, y0, erase_size_w,
                erase_size_h, x_erase, y_erase)
            weak_aug_video.append(w1)
            strong_aug_video.append(s1)
            weak_aug_bbox.append(wb1)
            strong_aug_bbox.append(sb1)
        return weak_aug_video, strong_aug_video, weak_aug_bbox, strong_aug_bbox, aug_probab_array
    else:
        for frame in range(video.shape[0]):
            # simple_frm, simple_bbox_aug = self.test_augs(video[frame], bbox_clip[frame])
            w1, wb1 = test_augs(video[frame], bbox_clip[frame], clip_h, clip_w, start_pos_h, start_pos_w)
            weak_aug_video.append(w1)
            weak_aug_bbox.append(wb1)
        return weak_aug_video, weak_aug_bbox, aug_probab_array

def get_basic_aug_views(video, bbox_clip, clip_h, clip_w, start_pos_h,
                                                   start_pos_w, modes, erase_size):
    aug_probab_array = np.random.rand(7)
    weak_aug_video = list()
    strong_aug_video = list()
    weak_aug_bbox = list()
    strong_aug_bbox = list()
    for frame in range(video.shape[0]):
        w1, s1, wb1, sb1 = train_basic_augs(video[frame], bbox_clip[frame], clip_h, clip_w, start_pos_h, start_pos_w)
        weak_aug_video.append(w1)
        strong_aug_video.append(s1)
        weak_aug_bbox.append(wb1)
        strong_aug_bbox.append(sb1)
    return weak_aug_video, strong_aug_video, weak_aug_bbox, strong_aug_bbox, aug_probab_array

def train_augs(frame, bbox_img, clip_h, clip_w, start_pos_h, start_pos_w, aug_probab_array, contrast_val,
                hue_val, saturation_val, brightness_val, gamma_val, cropping_factor, x0, y0, erase_size_w,
                erase_size_h, x_erase, y_erase):

    img = frame[start_pos_h:start_pos_h + 224, start_pos_w:start_pos_w + 224, :]
    bbox_img = bbox_img[start_pos_h:start_pos_h + 224, start_pos_w:start_pos_w + 224, :]
    img = cv2.resize(img, (clip_h, clip_w), interpolation=cv2.INTER_LINEAR)
    bbox_img = cv2.resize(bbox_img, (clip_h, clip_w), interpolation=cv2.INTER_LINEAR)
    simple_bbox = bbox_img > 0
    simple_bbox = simple_bbox.astype('uint8')
    orig_sum = np.sum(simple_bbox)
    # print(orig_sum)
    simple_bbox = simple_bbox * 255
    # print(np.sum(simple_bbox_aug))
    simple_bbox = np.expand_dims(simple_bbox, axis=2)
    # print(img.shape, img.dtype, img.max(), img.min())
    # print(simple_bbox_aug.shape, simple_bbox_aug.dtype)

    img = transforms.ToPILImage()(img)
    simple_bbox = transforms.ToPILImage()(simple_bbox)
    simple_frm = img

    # print(type(img), img.getextrema())
    # exit()

    if aug_probab_array[0] > 0.7:
        img = transforms.functional.adjust_contrast(img, contrast_val)
    if aug_probab_array[1] > 0.7:
        img = transforms.functional.adjust_hue(img, hue_val)
    if aug_probab_array[2] > 0.7:
        img = transforms.functional.adjust_brightness(img, brightness_val)
    if aug_probab_array[3] > 0.7:
        img = transforms.functional.adjust_saturation(img, saturation_val)
    if aug_probab_array[4] > 0.6:
        img = transforms.functional.to_grayscale(img, num_output_channels=3)
    if aug_probab_array[5] > 0.5:
        img = transforms.functional.gaussian_blur(img, kernel_size=(3, 3), sigma=(0.1, 2.0))

    # RIGHT NOW ST AND TEACHER ARE SEEING SAME GEOMETRICAL TRANSFORMATION
    # flip_probab = np.random.choice(2, 1)
    if aug_probab_array[6] > 0.5:
    # if flip_probab==1:
        img = transforms.functional.hflip(img)
        simple_frm = transforms.functional.hflip(simple_frm)

        simple_bbox_aug = transforms.functional.hflip(simple_bbox)
        strong_bbox_aug = transforms.functional.hflip(simple_bbox)

    else:

        simple_bbox_aug = simple_bbox
        strong_bbox_aug = simple_bbox

    # if aug_probab_array[7]>0.5:
    #     img = transforms.functional.resized_crop(img, y0, x0, int(224*cropping_factor), int(224*cropping_factor), (224, 224), interpolation=2)
    #     simple_frm = transforms.functional.resized_crop(simple_frm, y0, x0, int(224*cropping_factor), int(224*cropping_factor), (224, 224), interpolation=2)

    #     simple_bbox_aug = transforms.functional.resized_crop(simple_bbox_aug, y0, x0, int(224*cropping_factor), int(224*cropping_factor), (224, 224), interpolation=2)
    #     strong_bbox_aug = transforms.functional.resized_crop(strong_bbox_aug, y0, x0, int(224*cropping_factor), int(224*cropping_factor), (224, 224), interpolation=2)

    strong_frm = transforms.functional.to_tensor(img)
    simple_frm = transforms.functional.to_tensor(simple_frm)
    simple_bbox_aug = transforms.functional.to_tensor(simple_bbox_aug)
    strong_bbox_aug = transforms.functional.to_tensor(strong_bbox_aug)

    # if aug_probab_array[8] > 0.5:
    #     strong_frm = transforms.functional.erase(strong_frm, x_erase, y_erase, erase_size_h, erase_size_w, v=0) 
    #     strong_bbox_aug = transforms.functional.erase(strong_bbox_aug, x_erase, y_erase, erase_size_h, erase_size_w, v=0) 

    # assert orig_sum == torch.sum(simple_bbox_aug)
    return simple_frm, strong_frm, simple_bbox_aug, strong_bbox_aug

def train_basic_augs(frame, bbox_img, clip_h, clip_w, start_pos_h, start_pos_w):

    img = frame[start_pos_h:start_pos_h + 224, start_pos_w:start_pos_w + 224, :]
    bbox_img = bbox_img[start_pos_h:start_pos_h + 224, start_pos_w:start_pos_w + 224, :]
    img = cv2.resize(img, (clip_h, clip_w), interpolation=cv2.INTER_LINEAR)
    bbox_img = cv2.resize(bbox_img, (clip_h, clip_w), interpolation=cv2.INTER_LINEAR)
    simple_bbox = bbox_img > 0
    simple_bbox = simple_bbox.astype('uint8')
    orig_sum = np.sum(simple_bbox)
    simple_bbox = simple_bbox * 255
    simple_bbox = np.expand_dims(simple_bbox, axis=2)

    img = transforms.ToPILImage()(img)
    simple_bbox = transforms.ToPILImage()(simple_bbox)
    simple_frm = img

    # RIGHT NOW ST AND TEACHER ARE SEEING SAME GEOMETRICAL TRANSFORMATION
    flip_probab = np.random.choice(2, 1)
    if flip_probab==1:
        img = transforms.functional.hflip(img)
        simple_frm = transforms.functional.hflip(simple_frm)

        simple_bbox_aug = transforms.functional.hflip(simple_bbox)
        strong_bbox_aug = transforms.functional.hflip(simple_bbox)

    else:
        simple_bbox_aug = simple_bbox
        strong_bbox_aug = simple_bbox
    strong_frm = transforms.functional.to_tensor(img)
    simple_frm = transforms.functional.to_tensor(simple_frm)
    simple_bbox_aug = transforms.functional.to_tensor(simple_bbox_aug)
    strong_bbox_aug = transforms.functional.to_tensor(strong_bbox_aug)

    return simple_frm, strong_frm, simple_bbox_aug, strong_bbox_aug

def test_augs(frame, bbox_img, clip_h, clip_w, start_pos_h, start_pos_w):

    img = frame[start_pos_h:start_pos_h + 224, start_pos_w:start_pos_w + 224, :]
    bbox_img = bbox_img[start_pos_h:start_pos_h + 224, start_pos_w:start_pos_w + 224, :]
    img = cv2.resize(img, (clip_h, clip_w), interpolation=cv2.INTER_LINEAR)
    bbox_img = cv2.resize(bbox_img, (clip_h, clip_w), interpolation=cv2.INTER_LINEAR)
    simple_bbox = bbox_img > 0
    simple_bbox = simple_bbox.astype('uint8')
    orig_sum = np.sum(simple_bbox)
    simple_bbox = simple_bbox * 255
    simple_bbox = np.expand_dims(simple_bbox, axis=2)

    simple_frm = transforms.ToPILImage()(img)
    simple_bbox = transforms.ToPILImage()(simple_bbox)

    simple_frm = transforms.functional.to_tensor(simple_frm)
    simple_bbox_aug = transforms.functional.to_tensor(simple_bbox)

    return simple_frm, simple_bbox_aug


def get_multi_aug_views(frame, bbox_img, clip_h, clip_w, start_pos_h,
                                                   start_pos_w, modes, erase_size):
    aug_probab_array = np.random.rand(1, 7)

    cropping_factor = np.random.uniform(0.6, 1,
                                        size=(1,))  # on an average cropping factor is 80% i.e. covers 64% area
    x0 = [np.random.randint(0, 224 - 224 * cropping_factor[ii] + 1) for ii in range(1)]
    y0 = [np.random.randint(0, 224 - 224 * cropping_factor[ii] + 1) for ii in range(1)]

    x_erase = np.random.randint(0, 224 - erase_size, size=(1,))
    y_erase = np.random.randint(0, 224 - erase_size, size=(1,))
    erase_size_w = np.random.randint(int(erase_size / 2), erase_size, size=(1,))
    erase_size_h = np.random.randint(int(erase_size / 2), erase_size, size=(1,))

    contrast_val = np.random.uniform(0.6, 1.4, size=(1,))
    hue_val = np.random.uniform(-0.1, 0.1, size=(1,))
    saturation_val = np.random.uniform(0.6, 1.4, size=(1,))
    brightness_val = np.random.uniform(0.6, 1.4, size=(1,))
    gamma_val = np.random.uniform(0.6, 1.4, size=(1,))
    if modes=='train':
        return train_augs(frame, bbox_img, clip_h, clip_w, start_pos_h, start_pos_w, aug_probab_array, contrast_val,
                hue_val, saturation_val, brightness_val, gamma_val, cropping_factor, x0, y0, erase_size_w,
                erase_size_h, x_erase, y_erase)
    else:
        return test_augs(frame, bbox_img, clip_h, clip_w, start_pos_h, start_pos_w)

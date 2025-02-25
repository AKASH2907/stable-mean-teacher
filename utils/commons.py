import os
import sys
import random
import torch
import psutil
import imageio

import numpy as np
import os.path as osp

def init_seeds(seed=0, cuda_deterministic=True):
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



def visualize_pred_maps(path_dir, rgb_clips, index, filename):
    rgb_clips = rgb_clips.cpu().detach().numpy()
    rgb_clips = np.transpose(rgb_clips, [1, 2, 3, 0])

    with imageio.get_writer(osp.join(path_dir, '{}_{:02d}.gif'.format(filename, index)), mode='I') as writer:
        for i in range(rgb_clips.shape[0]):
            image = (rgb_clips[i] * 255).astype(np.uint8)
            writer.append_data(image)


def visualize_rgb_clips(path_dir, rgb_clips, index, filename):
    rgb_clips = rgb_clips.cpu().numpy()
    rgb_clips = np.transpose(rgb_clips, [1, 2, 3, 0])

    with imageio.get_writer(osp.join(path_dir, '{}_{:02d}.gif'.format(filename, index)), mode='I') as writer:
        for i in range(rgb_clips.shape[0]):
            image = (rgb_clips[i] * 255).astype(np.uint8)
            writer.append_data(image)


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory GB:', memoryUse)

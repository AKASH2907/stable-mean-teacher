import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import imageio
from utils.edge_detectors import DeepSobel


def update_ema(model, ema_model, global_step, alpha):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
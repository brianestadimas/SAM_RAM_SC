import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


# Calculate PSNR
def calculate_psnr(original_signal, roi_signal, theta=1.0):
    mse = torch.mean(torch.square(original_signal - roi_signal))
    
    signal_power = torch.square(torch.max(original_signal))
    noise_power = mse * theta # + mse_roni * (1 - theta)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def calculate_compression_rate(original_size, compressed_size):
    return compressed_size / original_size

# Calculate MIoU metrics
def calculate_miou(ground_truth, predicted):
    intersection = torch.logical_and(ground_truth, predicted)
    union = torch.logical_or(ground_truth, predicted)
    iou_score = torch.sum(intersection) / torch.sum(union)
    return iou_score

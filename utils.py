import math
import logging

import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt

from torchvision.transforms.functional import rotate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

def log_info(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f's{step}:{data}')
    
def circular_mask(h, w):
    r = min(h, w) // 2
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((Y - h//2)**2 + (X - w//2)**2)
    mask = d <= r
    return torch.from_numpy(mask.astype(np.float32))
    
def init_patch(im_dim, patch_r):
    patch_size = int(im_dim**2 * patch_r)
    r = int(math.sqrt(patch_size / math.pi))
    patch = np.zeros((r*2, r*2, 3))
    
    cx, cy = r, r
    y, x = np.ogrid[-r:r, -r:r]
    circ = x**2 + y**2 <= r**2
    
    for i in range(3):
        a = np.zeros((r*2, r*2))    

        init_values = np.random.rand(2*r, 2*r)
        a[cy-r:cy+r, cx-r:cx+r][circ] = init_values[cy-r:cy+r, cx-r:cx+r][circ]

        idx = np.flatnonzero((a == 0).all((1)))
        a = np.delete(a, idx, axis=0)
        patch[:, :, i] = np.delete(a, idx, axis=1)
        
    return torch.from_numpy(patch).float()

def transform(im_batch, patch, threshold=0.05):
    B, H, W, C = im_batch.shape
    PH, PW, _ = patch.shape
    
    # sample random rotation and location
    xs = np.random.uniform(0, W-PW, size=(B,)).astype(int)
    ys = np.random.uniform(0, H-PH, size=(B,)).astype(int)
    rots = np.random.uniform(-20, 20, size=(B,))

    p_batch = torch.zeros_like(im_batch)
    mask_batch = torch.zeros_like(im_batch)
    mask = circular_mask(PH, PW)
    
    for b, (x, y, rot) in enumerate(zip(xs, ys, rots)):
        rotated = rotate(patch.permute(2, 0, 1), rot)
        rotated = rotated.permute(1, 2, 0)
        rotated = torch.clamp(rotated, 0, 1)
        p_batch[b, y:y+PH, x:x+PW] = rotated
        mask_batch[b, y:y+PH, x:x+PW, :] = mask[:min(PH, H-y), :min(PW, W-x)].unsqueeze(-1)
    
    return p_batch, mask_batch.bool()
    
def apply_patch(im_batch, p_batch, mask):
    return torch.where(mask, p_batch, im_batch)

def imshow(im):
    plt.imshow(im.squeeze(0))

import math

import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt

from torchvision.transforms.functional import rotate

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
    
    for b, (x, y, rot) in enumerate(zip(xs, ys, rots)):
        rotated = rotate(patch.permute(2, 0, 1), rot)
        rotated = rotated.permute(1, 2, 0)
        rotated = torch.clamp(rotated, 0, 1)
        mask = (rotated > threshold).any(dim=-1)
        p_batch[b, y:y+PH, x:x+PW] = rotated
        mask_batch[b, y:y+PH, x:x+PW] = mask.unsqueeze(-1).repeat(1, 1, C)
    
    return p_batch, mask_batch.bool()
    
def apply_patch(im_batch, p_batch, mask):
    return torch.where(mask, p_batch, im_batch)

def imshow(im):
    plt.imshow(im.squeeze(0))

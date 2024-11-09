import os
import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import CLIPVisionModel, CLIPImageProcessor
from datasets import load_dataset
from tqdm import tqdm

from utils import init_patch, transform, apply_patch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_info(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f's{step}:{data}')

def imnet_loader(
    model_name='openai/clip-vit-large-patch14',
    split='train',
    batch_size=4,
    streaming=True,
    num_samples=None
):
    def preprocess(x, processor):
        return {'pixel_values': processor(x['image'])['pixel_values'][0], 'label': x['label']}
    
    processor = CLIPImageProcessor.from_pretrained(model_name)
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True).map(
        function=preprocess, fn_kwargs={'processor': processor}, remove_columns=['image'])
    if num_samples: dataset = dataset.shuffle(seed=42).take(num_samples)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True)

def patch_loader(
    split='train',
    batch_size=4,
    streaming=True,
    num_samples=None,
    target_label=0,
    **_
):
    def prepare_batch(x):
        img = np.array(x['image'])
        # check for greyscale images
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 1):
            img = np.stack([img] * 3, axis=-1)
        return {'pixel_values': torch.from_numpy(img).float() / 255.0, 'label': x['label']}

    def collate_fn(batch):
        return {'pixel_values': [x['pixel_values'] for x in batch], 'label': torch.stack([torch.tensor(x['label']) for x in batch])} 
    
    # this should happen lazily (?)
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True)\
        .map(function=prepare_batch, remove_columns=['image'])\
        .filter(lambda x: x['label'] != target_label)

    if num_samples: dataset = dataset.shuffle(seed=42).take(num_samples)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)

class CLIPClassifier(nn.Module):

    def __init__(
        self,
        model_name='openai/clip-vit-large-patch14',
        num_classes=1000,
        name=None,
        deep=None,
    ):
        super().__init__() 
           
        self.name = name
        self.num_classes = num_classes
        self.clip = CLIPVisionModel.from_pretrained(model_name)
        self.deep = deep

        self.hidden_size = self.clip.config.hidden_size
        for param in self.clip.parameters():
            param.requires_grad = False

        if deep is None:
            self.head = nn.Linear(self.hidden_size, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.hidden_size, deep),
                nn.ReLU(),
                nn.Linear(deep, self.num_classes)
            )
        
    def freeze(self):
        if self.deep is None:
            self.head.requires_grad = False
        else:
            for param in self.head:
                param.requires_grad = False
        
    def unfreeze(self):
        if self.deep is None:
            self.head.requires_grad = True
        else:
            for param in self.head:
                param.requires_grad = True
            
    def forward(self, inputs):
        h = self.clip(pixel_values=inputs['pixel_values']).last_hidden_state
        pooled = torch.mean(h[:, 1:, :], dim=1)
        logits = self.head(pooled)
        return logits

    def step(self, inputs):
        logits = self.forward(inputs)
        return F.cross_entropy(logits, inputs['label'])
    
    def save(self, path, step, optim):
        torch.save({'model': self.state_dict(), 'optim': optim.state_dict(), 'step': step}, path)
        
class Patch(nn.Module):

    def __init__(
        self,
        model,
        target_label,
        patch_r=0.05,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        patch=None,
        init_size=1024,
        name=None
    ):
        super().__init__()

        self.target_label = target_label
        self.patch_r = patch_r
        self.device = device
        self.name = name
        
        # we have to roll our own because CLIPImageProcessor breaks gradient flow 
        self.image_size = 224
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)

        # TODO -- for e2e this should be the entire VLM
        self.model = model
        self.model.freeze()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.patch = (init_patch(init_size, patch_r) if patch is None else patch).to(device)
        self.patch = nn.Parameter(self.patch, requires_grad=True)
        
    def process(self, img):
        img = F.interpolate(
            img.unsqueeze(0), 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        return (img - self.mean) / self.std
    
    def scale_patch(self, image_size):
        H, W = image_size
        PH, PW, _ = self.patch.shape
        
        A = H * W * self.patch_r
        scale = np.sqrt(A / (PH * PW))
        new_h = int(PH * scale)
        new_w = int(PW * scale)
        
        patch = self.patch.permute(2, 0, 1)
        patch = patch.unsqueeze(0)
        
        scaled = F.interpolate(
            patch,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )
        
        scaled = scaled.squeeze(0)
        return scaled.permute(1, 2, 0)
    
    def apply_patch(self, image):
        H, W, _ = image.shape
        scaled_patch = self.scale_patch((H, W))
        p_batch, mask = transform(image.unsqueeze(0), scaled_patch)
        patched = apply_patch(image.unsqueeze(0), p_batch, mask)
        patched = patched.squeeze(0).permute(2, 0, 1)  # H,W,C -> C,H,W
        return patched
    
    def forward(self, batch):
        patched = [self.apply_patch(img) for img in batch['pixel_values']]
        processed = torch.stack([self.process(img) for img in patched])
        return self.model({'pixel_values': processed, 'label': batch['label']})
    
    def step(self, batch):
        logits = self.forward(batch)
        targets = torch.full((logits.shape[0],), self.target_label, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, targets)
    
    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.model.eval()
        return self

@torch.no_grad() 
def val_classifier(model, val_loader, config, max_steps=None):
    model.eval()
    corr = n = 0
    
    for i, batch in enumerate(val_loader):
        if max_steps is not None and i >= max_steps: break
        inputs = {k: v.to(config['device']) for k, v in batch.items()}
        logits = model.forward(inputs)
        preds = torch.argmax(logits, dim=1)
        corr += (preds == inputs['label']).sum().item()
        n += inputs['label'].size()[0]
    
    return corr / n

def init(config):
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    model = CLIPClassifier(
        model_name=config['model_name'],
        num_classes=config.get('num_classes', 1000),
        name=datetime.now().strftime('%b%d_%H%M') + '_' + config.get('name', ''),
        deep=config.get('deep', False)
    ).to(config['device'])
    
    optim = AdamW(model.parameters(), lr=config['lr'])
    
    loader_params = {'model_name': config['model_name'], 'batch_size': config['batch_size'], 'streaming': config['streaming']}

    if config.get('patch'):
        train_loader = patch_loader(**loader_params, split='train', num_samples=config['num_train_samples'], target_label=config['target_label'])
        val_loader = patch_loader(**loader_params, split='validation', num_samples=config['num_val_samples'], target_label=config['target_label'])
    else:
        train_loader = imnet_loader(**loader_params, split='train', num_samples=config['num_train_samples'])
        val_loader = imnet_loader(**loader_params, split='validation', num_samples=config['num_val_samples'])

    return model, optim, train_loader, val_loader

def train_classifier(config):
    model, optim, train_loader, val_loader = init(config)
    logger.info('loaded')

    step = 0
    if config.get('resume_from'):
        checkpoint = torch.load(config['resume_from'], map_location=config['device'])
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        step = checkpoint['step']
        logger.info(f'resuming from {step}')
    
    val_classifier(model, val_loader, config, max_steps=1)
    logger.info('sanity check pass')

    for epoch in range(config['train_epochs']):
        logger.info(f'starting epoch {epoch + 1}')
        for batch in tqdm(train_loader):
            inputs = {k: v.to(config['device']) for k, v in batch.items()}

            model.train()
            loss = model.step(inputs)
            loss.backward()
            log_info({'train/loss': loss}, step=step)

            if (step + 1) % config['grad_accum_steps'] == 0:
                optim.step()
                optim.zero_grad()
            
            if (step + 1) % config['eval_at'] == 0:
                acc = val_classifier(model, val_loader, config)
                log_info({'eval/acc': acc}, step=step)
                path = Path(config['checkpoint_dir']) / f'imnet_best_model_{step}_{model.name}.pt'
                model.save(path, step, optim)
        
            step += 1
            
@torch.no_grad()
def val_patch(patch, val_loader, config, max_steps=None):
    patch.eval()
    
    corr = 0
    target_hits = 0
    n = 0
    
    for i, batch in tqdm(enumerate(val_loader)):
        if i >= max_steps: break
        batch = {'pixel_values': [t.to(config['device']) for t in batch['pixel_values']], 'label': batch['label'].to(config['device'])}
        logits = patch.forward(batch)
        preds = torch.argmax(logits, dim=-1)
        
        corr += (preds == batch['label']).sum()
        target_hits += (preds == config['target_label']).sum()
        n += batch['label'].size()[0]
    
    return corr / n, target_hits / n

@torch.no_grad()
def log_patch(patch, batch, step):
    patch_np = patch.patch.detach().cpu().numpy()
    img = batch['pixel_values'] [0]
    patched = patch.apply_patch(img).permute(1, 2, 0).cpu().detach().numpy()
    log_info({'patch': wandb.Image(patch_np), 'patched': wandb.Image(patched)}, step)

def train_patch(config):
    model, _, train_loader, val_loader = init(config)

    if config.get('model_from'):
        checkpoint = torch.load(config['model_from'], map_location=config['device'])
        model.load_state_dict(checkpoint['model'])
        logger.info('loaded pretrained classifier')
    
    patch = Patch(
        model=model,
        target_label=config['target_label'],
        device=config['device'],
        patch_r=config['patch_r'],
        name=config.get('name'),
        init_size=config.get('init_size', 1024)
    )

    optim = AdamW(patch.parameters(), lr=config['lr'])
    
    step = 0 
    if config.get('resume_patch_from'):
        checkpoint = torch.load(config['resume_patch_from'], map_location=config['device'])
        patch.patch = checkpoint['patch']
        step = checkpoint['step']
        logger.info(f'loaded patch from step: {step}')
        
    trainable_params = [n for n, p in patch.named_parameters() if p.requires_grad]
    assert len(trainable_params) == 1 and trainable_params[0] == 'patch'
   
    # logger.info('starting sanity check') 
    # val_patch(patch, val_loader, config, max_steps=1)
    # logger.info('passed!')
        
    for _ in range(config['train_epochs']):
        for batch in tqdm(train_loader):
            batch = {'pixel_values': [t.to(config['device']) for t in batch['pixel_values']], 'label': batch['label'].to(config['device'])}

            patch.train()
            loss = patch.step(batch)
            loss.backward()
            log_info({'train/loss': loss}, step=step)

            optim.step()
            optim.zero_grad()
            with torch.no_grad(): patch.patch.data.clamp_(0, 1)
            
            if (step + 1) % config['eval_at'] == 0:
                acc, success = val_patch(patch, val_loader, config)
                log_info({'eval/acc': acc, 'eval/success': success}, step=step)
                path = Path(config['checkpoint_dir']) / f'patch_{model.name}_{step}.pt'
                
            if (step + 1) % config['log_at'] == 0:
                log_patch(patch, batch, step)
            
            step += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', default='avlm')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    config['device'] = args.device
    config['patch'] = True # i genuinely have not figured out how removing this makes things break
    if args.wandb: wandb.init(project=args.wandb_project, config=config)
    if config.get('target_label'): train_patch(config)
    else: train_classifier(config)

if __name__ == '__main__':
    main()

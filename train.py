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

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# use this rather than directly calling logger
def log_info(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f's{step}:{data}')

def preprocess(x, processor):
    return {'pixel_values': processor(x['image'])['pixel_values'][0], 'label': x['label']}

def imnet_loader(
    model_name='openai/clip-vit-large-patch14',
    split='train',
    batch_size=4,
    streaming=True,
    num_workers=1,
    num_samples=None
):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming).map(
        function=preprocess, fn_kwargs={'processor': processor}, remove_columns=['image'])
    if num_samples: dataset = dataset.shuffle(seed=42).take(num_samples)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

def patch_loader():
    pass

def perturbation_loader():
    pass

class CLIPClassifier(nn.Module):
    '''
    simple classifier with a one-layer head and frozen clip weights
    applies mean pooling over last hidden state
    '''
    
    def __init__(
        self,
        model_name='openai/clip-vit-large-patch14',
        num_classes=1000,
        name=None
    ):
        super().__init__() 
           
        self.name = name
        self.num_classes = num_classes
        self.clip = CLIPVisionModel.from_pretrained(model_name)

        self.hidden_size = self.clip.config.hidden_size
        for param in self.clip.parameters():
            param.requires_grad = False

        self.head = nn.Linear(self.hidden_size, num_classes)
        
    def freeze(self):
        for param in self.head:
            param.requires_grad = False
        
    def unfreeze(self):
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
    
    def load(self, path, optim=None):
        checkpoint = torch.load(path, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint['model'])
        if optim: optim.load_state_dict(checkpoint['optim'])
       
@torch.no_grad() 
def val(model, val_loader, config, max_steps=None):
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
        name=datetime.now().strftime('%b%d_%H%M')).to(config['device'])
    
    optim = AdamW(model.parameters(), lr=config['lr'])

    train_loader = imnet_loader(
        model_name=config['model_name'],
        split='train',
        batch_size=config['batch_size'],
        streaming=config['streaming'],
        num_samples=config.get('num_train_samples', None))
    
    val_loader = imnet_loader(
        model_name=config['model_name'],
        split='validation',
        batch_size=config['batch_size'],
        streaming=config['streaming'],
        num_samples=config.get('num_val_samples', None))

    return model, optim, train_loader, val_loader

def train_classifier(config):
    model, optim, train_loader, val_loader = init(config)
    log_info('loaded')
    
    val(model, val_loader, config, max_steps=1)
    log_info('sanity check pass')

    step = best_acc = 0
    for epoch in range(config['train_epochs']):
        log_info(f'starting epoch {epoch + 1}')
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
                acc = val(model, val_loader, config)
                log_info({'eval/acc': acc}, step=step)
                if acc > best_acc:
                    best_acc = acc
                    path = Path(config['checkpoint_dir']) / f'imnet_best_model_{step}_{model.name}.pt'
                    model.save(path, step, optim)
            
            step += 1
    
def find_patch():
    pass

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
    if args.wandb: wandb.init(project=args.wandb_project, config=config)
    train_classifier(config)

if __name__ == '__main__':
    main()

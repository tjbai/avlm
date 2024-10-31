import argparse
import logging

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
    else: logger.info(f's{step}: {data}')

def preprocess(x, processor):
    return {'image': processor(x['image'])['pixel_values'][0], 'label': x['label']}

'''
TODO -- will need a separate patch attack loader that applies the
        patch during preprocessing
'''
def imnet_loader(
    model_name='openai/clip-vit-large-patch14',
    split='validation',
    batch_size=4,
    streaming=True,
    num_workers=1,
    num_samples=None
):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming).map(
        function=preprocess, fn_kwargs={'processor': processor})
    if num_samples: dataset = dataset.shuffle(seed=42).take(num_samples)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

class CLIPClassifier(nn.Module):
    '''
    simple classifier with a one-layer head and frozen clip weights
    applies mean pooling over last hidden state
    '''
    
    def __init__(
        self,
        model_name='openai/clip-vit-large-patch14',
        num_classes=1000
    ):
        super().__init__() 
            
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
        # TODO -- nll between logits and labels
        logits = self.forward(inputs)
       
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
        n += inputs['label'].size()
    
    return corr / n

def init(config):
    model = CLIPClassifier(
        model_name=config['model_name'],
        num_classes=config.get('num_classes', 1000)).to(config['device'])
    
    optim = AdamW(model.parameters(), lr=config['lr'])

    train_loader = imnet_loader(
        model_name=config['model_name'],
        split='train',
        batch_size=config['batch_size'],
        streaming=config['streaming'],
        num_workers=config['num_workers'],
        num_samples=config.get('num_train_samples', None))
    
    val_loader = imnet_loader(
        model_name=config['model_name'],
        split='val',
        batch_size=config['batch_size'],
        streaming=config['streaming'],
        num_workers=config['num_workers'],
        num_samples=config.get('num_val_samples', None))

    return model, optim, train_loader, val_loader

def train(config):
    model, optim, train_loader, val_loader = init(config)
    log_info('loaded') 
    
    val(model, val_loader, config, max_steps=1) 
    log_info('sanity check pass') 

    step = 0
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
            
            step += 1
    
def find_patch():
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == '__main__':
    main()

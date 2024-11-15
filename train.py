import os
import yaml
import argparse
import logging
import contextlib
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from torch.optim import AdamW
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import CLIPVisionModel
from tqdm import tqdm

from attack import Patch
from data import imnet_loader, patch_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

def log_info(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f's{step}:{data}')

def log_profiler(prof, step):
    if prof is None: return

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    logger.info(f"profiler at step{step}:\n{table}")
    
    if wandb.run is not None:
        events = [{
            'name': item.key,
            'cuda_time_ms': item.cuda_time_total / 1000,
            'cpu_time_ms': item.cpu_time_total / 1000,
            'calls': item.count
        } for item in prof.key_averages()]

        wandb.log({
            "profiler/top_operations": wandb.Table(data=events),
            "profiler/memory_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            "profiler/memory_reserved_gb": torch.cuda.max_memory_reserved() / 1e9
        }, step=step)
        
@contextlib.contextmanager
def create_profiler(enabled=False):
    if enabled:
        profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA,],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    else:
        profiler = None

    try:
        yield profiler
    finally:
        if profiler is not None:
            profiler.stop()

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
    
    loader_params = {'model_name': config['model_name'], 'batch_size': config['batch_size'], 'streaming': config['streaming'], 'num_workers': config.get('num_workers', 4)}

    if config.get('attack_type') == 'patch':
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
                optim.zero_grad(set_to_none=True)
            
            if (step + 1) % config['eval_at'] == 0:
                acc = val_classifier(model, val_loader, config)
                log_info({'eval/acc': acc}, step=step)
                path = Path(config['checkpoint_dir']) / f'imnet_best_model_{step}_{model.name}.pt'
                model.save(path, step, optim)
        
            step += 1
            
def train_attack(config):
    model, _, train_loader, val_loader = init(config)

    if config.get('model_from'):
        checkpoint = torch.load(config['model_from'], map_location=config['device'])
        model.load_state_dict(checkpoint['model'])
        
    kwargs = {'device': config['device'], 'target_label': config['target_label'], 'name': config['name']}
    attack = Patch(model, **kwargs, patch_r=config['patch_r'], init_size=config['init_size'])
    optim = AdamW(attack.trainable_params(), lr=config['lr'])
    
    step = 0
    if config.get('resume_from'):
        checkpoint = torch.load(config['resume_from'], map_location=config['device'])
        optim.load_state_dict(checkpoint['optim'])
        attack.load_params(checkpoint['params'])
        step = checkpoint['step']
        logger.info(f'loaded attack from step: {step}')
    
    logger.info('eval sanity check...')
    attack.val_attack(val_loader, config, max_steps=1)
    logger.info('passed!')

    for _ in range(config['train_epochs']):
        # with create_profiler(enabled=config.get('profile', False)) as prof:
        for batch in tqdm(train_loader):
            batch = {k: v.to(config['device']) for k, v in batch.items()}

            try:
                with record_function('training_step'):
                    attack.train()
                    loss = attack.step(batch)
                    loss.backward()
                    log_info({'train/loss': loss}, step=step)
                    attack.pre_update(optim)
                    optim.step()
                    optim.zero_grad()
                    attack.post_update(optim)

            except RuntimeError as e:
                logger.info(f'encountered an error at step={step}:\n{e}')
            
            if (step + 1) % config['eval_at'] == 0:
                acc, success = attack.val_attack(val_loader, config)
                log_info({'eval/acc': acc, 'eval/success': success}, step=step)
                path = Path(config['checkpoint_dir']) / f'attack_{attack.name}_{step}.pt'
                attack.save(path, optim, step)
            
            if (step + 1) % config['log_at'] == 0: attack.log_patch(batch, step)

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
    train_attack(config)

if __name__ == '__main__':
    main()

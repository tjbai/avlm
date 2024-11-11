import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from abc import ABC, abstractmethod
from tqdm import tqdm
from utils import init_patch, transform, apply_patch, log_info

class Attack(nn.Module, ABC):

    def __init__(
        self,
        model,
        target_label,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        name=None
    ):
        super().__init__()
        self.model = model
        self.target_label = target_label
        self.device = device
        self.name = name
        
        # processing params
        self.image_size = 224
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(self.device)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    @abstractmethod
    def trainable_params(self):
        pass

    @abstractmethod
    def load_params(self):
        pass
    
    @abstractmethod 
    def apply_attack(self, images):
        '''process and/or apply attack to input images'''
        pass

    @abstractmethod
    def pre_update(self, optim):
        '''pre-update hook to access gradients, etc.'''
        pass
   
    @abstractmethod
    def post_update(self, optim):
        '''post-update hook for clamping/projection'''
        pass
    
    @abstractmethod
    def val_attack(self, val_loader, config, max_steps=None):
        '''should return accuracy and misclassification rate''' 
        pass
    
    @abstractmethod
    def log_patch(self, batch, step):
        '''log params/attacked images to console/wandb for observability'''
        pass
        
    def forward(self, batch):
        processed = self.apply_attack(batch['pixel_values'])
        return self.model({'pixel_values': processed, 'label': batch['label']})
    
    def criterion(self, logits):
        targets = torch.full((logits.shape[0],), self.target_label, dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, targets)
    
    def step(self, batch):
        logits = self.forward(batch)
        return self.criterion(logits)

    def save(self, path, optim, step):
        torch.save({'params': self.trainable_params(), 'optim': optim.state_dict(), 'step': step}, path)

class Patch(Attack):

    def __init__(
        self,
        model: nn.Module,
        target_label: int,
        patch_r=0.05,
        init_size=1024,
        patch=None,
        **kwargs
    ):
        super().__init__(model, target_label, **kwargs)
        self.patch_r = patch_r
        self.patch = (init_patch(init_size, patch_r) if patch is None else patch).to(self.device)
        self.patch = nn.Parameter(self.patch, requires_grad=True)
    
    def trainable_params(self):
        return [self.patch]
        
    def _process(self, img):
        img = F.interpolate(
            img.unsqueeze(0), 
            size=(self.image_size, self.image_size), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)

        return (img - self.mean) / self.std
    
    def _scale_patch(self, img_size):
        H, W = img_size
        PH, PW, _ = self.patch.shape
        
        A = H * W * self.patch_r
        scale = np.sqrt(A / (PH * PW))
        new_h = int(PH * scale)
        new_w = int(PW * scale)
        
        patch = self.patch.permute(2, 0, 1).unsqueeze(0)
        scaled = F.interpolate(patch, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return scaled.squeeze(0).permute(1, 2, 0)
    
    def _apply_patch(self, img):
        H, W, _ = img.shape
        scaled_patch = self._scale_patch((H, W))
        p_batch, mask = transform(img.unsqueeze(0), scaled_patch)
        patched = apply_patch(img.unsqueeze(0), p_batch, mask)
        patched = patched.squeeze(0).permute(2, 0, 1) # H,W,C -> C,H,W
        return patched
    
    def apply_attack(self, imgs):
        patched = [self._apply_patch(img) for img in imgs]
        return torch.stack([self._process(img) for img in patched])
    
    def pre_update(self, *_, **__):
        pass
    
    def post_update(self, *_, **__):
        with torch.no_grad():
            self.patch.data.clamp_(0, 1)
        
    def load_params(self, params):
        self.patch = params[0]
    
    @torch.no_grad() 
    def val_attack(self, val_loader, config, max_steps=None):
        self.eval()
        
        corr = 0
        target_hits = 0
        n = 0
        
        for i, batch in tqdm(enumerate(val_loader)):
            if max_steps is not None and i >= max_steps: break
            batch = {'pixel_values': [t.to(config['device']) for t in batch['pixel_values']], 'label': batch['label'].to(config['device'])}
            logits = self.forward(batch)
            preds = torch.argmax(logits, dim=-1)
            
            corr += (preds == batch['label']).sum()
            target_hits += (preds == config['target_label']).sum()
            n += batch['label'].size()[0]
        
        return corr / n, target_hits / n

    @torch.no_grad()
    def log_patch(self, batch, step):
        patch_np = self.patch.detach().cpu().numpy()
        img = batch['pixel_values'] [0]
        patched = self._apply_patch(img).permute(1, 2, 0).cpu().detach().numpy()
        log_info({'patch': wandb.Image(patch_np), 'patched': wandb.Image(patched)}, step)

# class Patch(nn.Module):

#     def __init__(
#         self,
#         model,
#         target_label,
#         patch_r=0.05,
#         device='cuda' if torch.cuda.is_available() else 'cpu',
#         patch=None,
#         init_size=1024,
#         name=None
#     ):
#         super().__init__()

#         self.target_label = target_label
#         self.patch_r = patch_r
#         self.device = device
#         self.name = name
        
#         # we have to roll our own because CLIPImageProcessor breaks gradient flow 
#         self.image_size = 224
#         self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
#         self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)

#         # TODO -- for e2e this should be the entire VLM
#         self.model = model
#         self.model.eval()
#         for param in self.model.parameters():
#             param.requires_grad = False
            
#         self.patch = (init_patch(init_size, patch_r) if patch is None else patch).to(device)
#         self.patch = nn.Parameter(self.patch, requires_grad=True)
        
#     def process(self, img):
#         img = F.interpolate(
#             img.unsqueeze(0), 
#             size=(self.image_size, self.image_size), 
#             mode='bilinear', 
#             align_corners=False
#         ).squeeze(0)
        
#         return (img - self.mean) / self.std
    
#     def scale_patch(self, image_size):
#         H, W = image_size
#         PH, PW, _ = self.patch.shape
        
#         A = H * W * self.patch_r
#         scale = np.sqrt(A / (PH * PW))
#         new_h = int(PH * scale)
#         new_w = int(PW * scale)
        
#         patch = self.patch.permute(2, 0, 1)
#         patch = patch.unsqueeze(0)
        
#         scaled = F.interpolate(
#             patch,
#             size=(new_h, new_w),
#             mode='bilinear',
#             align_corners=False
#         )
        
#         scaled = scaled.squeeze(0)
#         return scaled.permute(1, 2, 0)
    
#     def apply_patch(self, image):
#         H, W, _ = image.shape
#         scaled_patch = self.scale_patch((H, W))
#         p_batch, mask = transform(image.unsqueeze(0), scaled_patch)
#         patched = apply_patch(image.unsqueeze(0), p_batch, mask)
#         patched = patched.squeeze(0).permute(2, 0, 1)  # H,W,C -> C,H,W
#         return patched
    
#     def forward(self, batch):
#         patched = [self.apply_patch(img) for img in batch['pixel_values']]
#         processed = torch.stack([self.process(img) for img in patched])
#         return self.model({'pixel_values': processed, 'label': batch['label']})
    
#     def step(self, batch):
#         logits = self.forward(batch)
#         targets = torch.full((logits.shape[0],), self.target_label, dtype=torch.long, device=self.device)
#         return F.cross_entropy(logits, targets)
    
#     def train(self, *args, **kwargs):
#         super().train(*args, **kwargs)
#         self.model.eval()
#         return self
    
#     def save(self, path, step, optim):
#         torch.save({'patch': self.patch, 'optim': optim.state_dict(), 'step': step}, path)

# @torch.no_grad()
# def val_patch(patch, val_loader, config, max_steps=None):
#     patch.eval()
    
#     corr = 0
#     target_hits = 0
#     n = 0
    
#     for i, batch in tqdm(enumerate(val_loader)):
#         if max_steps is not None and i >= max_steps: break
#         batch = {'pixel_values': [t.to(config['device']) for t in batch['pixel_values']], 'label': batch['label'].to(config['device'])}
#         logits = patch.forward(batch)
#         preds = torch.argmax(logits, dim=-1)
        
#         corr += (preds == batch['label']).sum()
#         target_hits += (preds == config['target_label']).sum()
#         n += batch['label'].size()[0]
    
#     return corr / n, target_hits / n

# @torch.no_grad()
# def log_patch(patch, batch, step):
#     patch_np = patch.patch.detach().cpu().numpy()
#     img = batch['pixel_values'] [0]
#     patched = patch.apply_patch(img).permute(1, 2, 0).cpu().detach().numpy()
#     log_info({'patch': wandb.Image(patch_np), 'patched': wandb.Image(patched)}, step)

# def train_patch(config):
#     model, _, train_loader, val_loader = init(config)

#     if config.get('model_from'):
#         checkpoint = torch.load(config['model_from'], map_location=config['device'])
#         model.load_state_dict(checkpoint['model'])
#         logger.info('loaded pretrained classifier')
    
#     patch = Patch(
#         model=model,
#         target_label=config['target_label'],
#         device=config['device'],
#         patch_r=config['patch_r'],
#         name=config.get('name'),
#         init_size=config.get('init_size', 1024)
#     )

#     optim = AdamW(patch.parameters(), lr=config['lr'])
    
#     step = 0
#     if config.get('resume_patch_from'):
#         checkpoint = torch.load(config['resume_patch_from'], map_location=config['device'])
#         patch.patch = checkpoint['patch']
#         step = checkpoint['step']
#         logger.info(f'loaded patch from step: {step}')
        
#     trainable_params = [n for n, p in patch.named_parameters() if p.requires_grad]
#     assert len(trainable_params) == 1 and trainable_params[0] == 'patch'
   
#     logger.info('starting sanity check') 
#     val_patch(patch, val_loader, config, max_steps=1)
#     logger.info('passed!')
        
#     for _ in range(config['train_epochs']):
#         for batch in tqdm(train_loader):
#             batch = {'pixel_values': [t.to(config['device']) for t in batch['pixel_values']], 'label': batch['label'].to(config['device'])}

#             patch.train()
#             loss = patch.step(batch)
#             loss.backward()
#             log_info({'train/loss': loss}, step=step)

#             optim.step()
#             optim.zero_grad()
#             with torch.no_grad(): patch.patch.data.clamp_(0, 1)
            
#             if (step + 1) % config['eval_at'] == 0:
#                 acc, success = val_patch(patch, val_loader, config)
#                 log_info({'eval/acc': acc, 'eval/success': success}, step=step)
#                 path = Path(config['checkpoint_dir']) / f'patch_{model.name}_{step}.pt'
#                 patch.save(path, step, optim)
                
#             if (step + 1) % config['log_at'] == 0:
#                 log_patch(patch, batch, step)
            
#             step += 1

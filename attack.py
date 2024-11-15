import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(self.device)
        # self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(self.device)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 1, 3).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 1, 3).to(self.device)
        
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
        with torch.cuda.amp.autocast():
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
        model,
        target_label,
        patch_r=0.05,
        init_size=224,
        patch=None,
        **kwargs
    ):
        super().__init__(model, target_label, **kwargs)
        self.patch_r = patch_r
        self.patch = (init_patch(init_size, patch_r) if patch is None else patch).to(self.device)
        self.patch = nn.Parameter(self.patch, requires_grad=True)
    
    def trainable_params(self):
        return [self.patch]
    
    def _apply_patch(self, imgs):
        p_batch, mask = transform(imgs, self.patch)
        return apply_patch(imgs, p_batch, mask)
        
    def _process(self, imgs):
        return (imgs - self.mean) / self.std
    
    def apply_attack(self, imgs):
        return self._process(self._apply_patch(imgs)).permute(0, 3, 1, 2)
    
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
            batch = {k: v.to(config['device']) for k , v in batch.items()}
            logits = self.forward(batch)
            preds = torch.argmax(logits, dim=-1)
            
            corr += (preds == batch['label']).sum()
            target_hits += (preds == config['target_label']).sum()
            n += batch['label'].size()[0]
        
        return corr / n, target_hits / n

    @torch.no_grad()
    def log_patch(self, batch, step):
        patch_np = self.patch.detach().cpu().numpy()
        patched = self._apply_patch(batch['pixel_values'])[0].cpu().detach().numpy()
        log_info({'patch': wandb.Image(patch_np), 'patched': wandb.Image(patched)}, step)

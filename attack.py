import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from abc import ABC, abstractmethod
from tqdm import tqdm
from utils import init_patch, transform, apply_patch, log_info

class Attack(nn.Module, ABC):

    def __init__(
        self,
        model=None,
        target_label=None,
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
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 1, 3).to(self.device)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 1, 3).to(self.device)
       
        if self.model:
            self.model.eval()
            # for param in self.model.parameters():
            #     param.requires_grad = False
            
    def trainable_params(self):
        raise NotImplementedError()

    def load_params(self):
        raise NotImplementedError()
    
    def apply_attack(self, images, normalize=True):
        raise NotImplementedError()

    def pre_update(self, optim):
        '''pre-update hook to access gradients, etc.'''
        raise NotImplementedError()
   
    def post_update(self, optim):
        '''post-update hook for clamping/projection'''
        raise NotImplementedError()
    
    def val_attack(self, val_loader, config, max_steps=None):
        '''should return accuracy and misclassification rate''' 
        raise NotImplementedError()
    
    def log_patch(self, batch, step):
        '''log params/attacked images to console/wandb for observability'''
        raise NotImplementedError()
        
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
        
    def normalize(self, imgs):
        return (imgs - self.mean) / self.std
        
class Identity(Attack):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def apply_attack(self, imgs, **_):
        return imgs.permute(0, 3, 1, 2)
    
    def load_params(self, *_, **__):
        return

class Patch(Attack):

    def __init__(
        self,
        model=None,
        target_label=None,
        patch_r=0.05,
        init_size=224,
        patch=None,
        **kwargs
    ):
        super().__init__(model, target_label, **kwargs)
        self.patch_r = patch_r
        self.patch = (init_patch(init_size) if patch is None else patch).to(self.device)
        self.patch = nn.Parameter(self.patch, requires_grad=True)

        # precompute downscaled radius
        A = int(224**2 * patch_r)
        r = int(math.sqrt(A / math.pi))
        self.resize = torchvision.transforms.Resize((2*r, 2*r))
    
    def trainable_params(self):
        return [self.patch]
    
    def _apply_patch(self, imgs):
        patch = F.sigmoid(self.patch)
        patch = self.resize(patch.permute(2, 0, 1)).permute(1, 2, 0)
        p_batch, mask = transform(imgs, patch)
        return apply_patch(imgs, p_batch, mask)
    
    def apply_attack(self, imgs, normalize=True):
        res = self._apply_patch(imgs)
        if normalize: res = self.normalize(res)
        return res.permute(0, 3, 1, 2)
    
    def pre_update(self, *_, **__):
        pass
    
    def post_update(self, *_, **__):
        pass
        
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
        patch_np = F.sigmoid(self.patch).detach().cpu().numpy()
        patched = self._apply_patch(batch['pixel_values'])[0].cpu().detach().numpy()
        log_info({'patch': wandb.Image(patch_np), 'patched': wandb.Image(patched)}, step)

class FGSM(Attack):
    def __init__(self, model, target_label, epsilon=0.03, **kwargs):
        super().__init__(model, target_label, **kwargs)
        self.epsilon = epsilon

    def trainable_params(self):
        return []

    def load_params(self, params):
        pass
    def apply_attack(self, images):
        images = images.permute(0, 3, 1, 2).requires_grad_(True)
        images = images.detach()
        images = images.requires_grad_(True)
        outputs = self.model({'pixel_values': images})
        loss = self.criterion(outputs)
        self.model.zero_grad()
        loss.backward()
        assert images.grad is not None, "Gradients w.r.t. images are not being computed."
        perturbation = self.epsilon * images.grad.sign()
        adversarial_images = images + perturbation
        return torch.clamp(adversarial_images, 0, 1).detach()
    def pre_update(self, *_, **__):
        pass 

    def post_update(self, *_, **__):
        pass 

    def val_attack(self, val_loader, config, max_steps=None):
        self.eval()
        corr = target_hits = n = 0

        for i, batch in tqdm(enumerate(val_loader)):
            if max_steps is not None and i >= max_steps:
                break
            batch = {k: v.to(config['device']) for k, v in batch.items()}
            batch['pixel_values'] = batch['pixel_values'].requires_grad_(True)
            # with torch.enable_grad():
            adv_images = self.apply_attack(batch['pixel_values'])
            logits = self.model({'pixel_values': adv_images})
            preds = torch.argmax(logits, dim=-1)
            corr += (preds == batch['label']).sum().item()
            target_hits += (preds == config['target_label']).sum().item()
            n += batch['label'].size()[0]

            batch = {k: v.to(config['device']) for k, v in batch.items()}
            adv_images = self.apply_attack(batch['pixel_values'])
            outputs = self.model({'pixel_values': adv_images})
            preds = torch.argmax(outputs, dim=-1)
            success_rate = (preds == self.target_label).float().mean()
            print(f"Attack Success Rate: {success_rate.item() * 100:.2f}%")


        return corr / n, target_hits / n

    def log_patch(self, batch, step):
        adv_images = self.apply_attack(batch['pixel_values'])[0]
        log_info({'image': wandb.Image(batch['pixel_values'][0].cpu().detach().numpy()), 'pertrubed': wandb.Image(adv_images)}, step)


    def hook_fn(grad):
        print("Gradient rec by images:", grad)

class PGD(Attack):
    def __init__(self, model, target_label, epsilon=0.03, alpha=0.005, num_steps=10, **kwargs):
        super().__init__(model, target_label, **kwargs)
        self.epsilon = epsilon 
        self.alpha = alpha 
        self.num_steps = num_steps

    def trainable_params(self):
        return []

    def load_params(self, params):
        pass

    def apply_attack(self, images):
        original_images = images.clone().detach()

        if images.shape[-1] == 3: 
            images = images.permute(0, 3, 1, 2)
            images = images.detach()
            images = images.requires_grad_(True)
        if original_images.shape[-1] == 3: 
            original_images = original_images.permute(0, 3, 1, 2) 
        for step in range(self.num_steps):
            images.requires_grad_(True)
            outputs = self.model({'pixel_values': images})
            loss = self.criterion(outputs)
            self.model.zero_grad()
            loss.backward()
            perturbation = self.alpha * images.grad.sign()
            images = images + perturbation
            images = torch.clamp(images, original_images - self.epsilon, original_images + self.epsilon)
            images = torch.clamp(images, 0, 1)
            images = images.detach()
        return images

    def pre_update(self, *_, **__):
        pass 

    def post_update(self, *_, **__):
        pass 
    
    def val_attack(self, val_loader, config, max_steps=None):
        self.eval()
        corr = target_hits = n = 0

        for i, batch in tqdm(enumerate(val_loader)):
            if max_steps is not None and i >= max_steps:
                break

            batch = {k: v.to(config['device']) for k, v in batch.items()}
            batch['pixel_values'] = batch['pixel_values'].clone().requires_grad_(True)
            # with torch.enable_grad():
            adv_images = self.apply_attack(batch['pixel_values'])
            logits = self.model({'pixel_values': adv_images})
            preds = torch.argmax(logits, dim=-1)

            corr += (preds == batch['label']).sum().item()
            target_hits += (preds == config['target_label']).sum().item()
            n += batch['label'].size()[0]

        return corr / n, target_hits / n

    def log_patch(self, batch, step):
        pass
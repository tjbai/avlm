import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
from datasets import load_dataset

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
        num_classes=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__() 
            
        self.device = device 
        self.num_classes = num_classes
        self.clip = CLIPVisionModel.from_pretrained(model_name).to(device)

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
            
    def forward(self, inputs, is_processed=True):
        h = self.clip(pixel_values=inputs['pixel_values']).last_hidden_state
        pooled = torch.mean(h[:, 1:, :], dim=1)
        logits = self.head(pooled)
        return logits

    def step(self, im_batch, labels):
        # TODO -- nll between logits and labels
        logits = self.forward(im_batch)
       
@torch.no_grad() 
def val(model, loader, config):
    model.eval()
    corr = n = 0
    
    for batch in loader:
        pass

def train_clip(
    model,
    train_loader,
    config
):
    pass

def find_patch():
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == '__main__':
    main()

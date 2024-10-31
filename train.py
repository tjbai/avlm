import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import CLIPVisionModel, CLIPImageProcessor, LlavaForConditionalGeneration

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
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

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
            
    def forward(self, im_batch):
        inputs = self.processor(im_batch, do_rescale=False, return_tensors='pt')
        h = self.clip(**inputs).last_hidden_state
        pooled = torch.mean(h[:, 1:, :], dim=1)
        logits = self.head(pooled)
        return logits
    
    def step(self, im_batch, labels):
        # TODO -- nll between logits and labels
        logits = self.forward(im_batch) 

# step 1: train classifier on imagenet. clip frozen.
def train_clip():
    pass

# step 2: train patch under EoT. clip and classifier frozen.
def train_patch():
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def main():
    args = parse_args()

if __name__ == '__main__':
    main()

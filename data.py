# imagenet or cifar?

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

def imnet_loader(bsz=4, split='validation', streaming=True):
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming)

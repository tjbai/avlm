# imagenet or cifar?

from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

def imnet_loader(split='validation', batch_size=4, streaming=True, num_workers=1):
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)



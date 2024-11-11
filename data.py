import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor
from datasets import load_dataset

def imnet_loader(
    model_name='openai/clip-vit-large-patch14',
    split='train',
    batch_size=4,
    streaming=True,
    num_samples=None
):
    def preprocess(x, processor):
        return {'pixel_values': processor(x['image'])['pixel_values'][0], 'label': x['label']}
    
    processor = CLIPImageProcessor.from_pretrained(model_name)
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True).map(
        function=preprocess, fn_kwargs={'processor': processor}, remove_columns=['image'])
    if num_samples: dataset = dataset.shuffle(seed=42).take(num_samples)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True)

def patch_loader(
    split='train',
    batch_size=4,
    streaming=True,
    num_samples=None,
    target_label=0,
    **_
):
    def prepare_batch(x):
        img = np.array(x['image'])
        # check for greyscale images
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 1):
            img = np.stack([img] * 3, axis=-1)
        return {'pixel_values': torch.from_numpy(img).float() / 255.0, 'label': x['label']}

    def collate_fn(batch):
        return {'pixel_values': [x['pixel_values'] for x in batch], 'label': torch.stack([torch.tensor(x['label']) for x in batch])}
    
    # this should happen lazily (?)
    dataset = load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True)\
        .map(function=prepare_batch, remove_columns=['image'])\
        .filter(lambda x: x['label'] != target_label)

    if num_samples: dataset = dataset.shuffle(seed=42).take(num_samples)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)

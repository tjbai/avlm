import io
import os
import tarfile
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from transformers import CLIPImageProcessor
from datasets import load_dataset
from datasets import IterableDataset
from torchvision import transforms

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

def prepare_batch(x, transform=None):
    return {'pixel_values': transform(x['image']), 'label': torch.tensor(x['label'], dtype=torch.long)}

def filter_target(x, target_label=0):
    return x['label'] != target_label

def adjust_greyscale(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

def permute(x):
    return x.permute(1, 2, 0)

def patch_loader(
    split='train',
    batch_size=4,
    streaming=True,
    target_label=0,
    **_
):
    if streaming:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(adjust_greyscale),
            transforms.Lambda(permute)
        ])

        dataset = load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True)\
            .map(function=prepare_batch, fn_kwargs={'transform': transform}, remove_columns=['image'])\
            .filter(function=filter_target, fn_kwargs={'target_label': target_label})

    else:
        # dataset = DIYImagenet('./data', split=split, target_label=target_label)
        dataset = DIYImagenet('/scratch4/jeisner1/imnet_files/data', split=split, target_label=target_label)

    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=0)

class DIYImagenet(IterableDataset):

    def __init__(self, tar_dir, split='train', target_label=0):
        self._epoch = 0
        self.dataset = iter_imnet(tar_dir, split=split)
        self.target_label = target_label
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(adjust_greyscale),
            transforms.Lambda(permute)
        ])

    def __iter__(self):
        for item in self.dataset:
            label = int(item['label'])
            if label == self.target_label: continue
            img = Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB')
            yield {'pixel_values': self.transform(img), 'label': torch.tensor(label, dtype=torch.long)}
            
def gen_imnet(paths=None, to_label=None):
    for tar_path in paths:
        with tarfile.open(tar_path, 'r:gz') as archive:
            for member in archive:
                if member.name.endswith('.JPEG'):
                    f = archive.extractfile(member)
                    if f is not None:
                        root, _ = os.path.splitext(member.name)
                        _, synset_id = os.path.basename(root).rsplit("_", 1)
                        yield {"image": {"path": member.name, "bytes": f.read()}, "label": to_label[synset_id]}

def iter_imnet(tar_dir, split='train'):
    from classes import IMAGENET2012_CLASSES
    to_label = {s: i for i, s in enumerate(IMAGENET2012_CLASSES.keys())}

    tars = {
        'train': [
            os.path.join(tar_dir, 'train_images_0.tar.gz'),
            os.path.join(tar_dir, 'train_images_1.tar.gz'),
            os.path.join(tar_dir, 'train_images_2.tar.gz'),
            os.path.join(tar_dir, 'train_images_3.tar.gz')
        ],
        'validation': [os.path.join(tar_dir, 'val_images.tar.gz')],
        'test': [os.path.join(tar_dir, 'test_images.tar.gz')],
    }

    return IterableDataset.from_generator(gen_imnet, gen_kwargs={'paths': tars.get(split), 'to_label': to_label})

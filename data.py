import torch
import numpy as np

from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor
from datasets import load_dataset, load_from_disk

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

def prepare_batch(x):
    img = np.array(x['image'])
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[-1] == 1): img = np.stack([img] * 3, axis=-1)
    return {'pixel_values': torch.from_numpy(img).float() / 255.0, 'label': torch.tensor(x['label'], dtype=torch.long)}

def collate_fn(batch):
    return {'pixel_values': [x['pixel_values'] for x in batch], 'label': torch.stack([x['label'] for x in batch])}

def filter_class(x, target_label=0): return x['label'] != target_label

def patch_loader(
    split='train',
    batch_size=4,
    streaming=True,
    num_samples=None,
    target_label=0,
    num_workers=4,
    **_
):
    if streaming:
        dataset = load_dataset('imagenet-1k', split=split, streaming=streaming, trust_remote_code=True)\
            .map(function=prepare_batch, remove_columns=['image'])\
            .filter(function=filter_class, fn_kwargs={'target_label': target_label})
    else:
        # import os
        # os.environ['HF_DATASETS_CACHE'] = '/scratch4/jeisner1/imnet'
        # from datasets import config
        # print(f'caching to {config.HF_DATASETS_CACHE}')

        # dataset = load_dataset('/scratch4/jeisner1/imnet', split=split)\
        dataset = load_dataset('imagenet-1k', split=split, trust_remote_code=True, cache_dir='/scratch4/jeisner1/imnet')\
             .map(function=prepare_batch, remove_columns=['image'], num_proc=num_workers)\
             .filter(function=filter_class, fn_kwargs={'target_label': target_label})

    if num_samples: dataset = dataset.shuffle(seed=42).take(num_samples)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn, num_workers=num_workers)

from datasets import IterableDataset, Dataset, DatasetInfo, Features, ClassLabel, Image
import tarfile
import io
import os

def iter_imnet(tar_dir):
    """
    Create an iterable dataset from ImageNet tar.gz files using streaming
    """
    from classes import IMAGENET2012_CLASSES

    train_tars = [
        os.path.join(tar_dir, 'train_images_0.tar.gz'),
        os.path.join(tar_dir, 'train_images_1.tar.gz')
    ]

    def gen():
        for tar_path in train_tars:
            with tarfile.open(tar_path, 'r:gz') as archive:
                for member in archive:
                    if member.name.endswith('.JPEG'):
                        f = archive.extractfile(member)
                        if f is not None:
                            # Extract synset ID from filename
                            root, _ = os.path.splitext(member.name)
                            _, synset_id = os.path.basename(root).rsplit("_", 1)

                            yield {
                                "image": {"path": member.name, "bytes": f.read()},
                                "label": IMAGENET2012_CLASSES[synset_id]
                            }

    return IterableDataset.from_generator(gen)

def load_imnet(tar_dir):
    """
    Load ImageNet from local tar.gz files
    """
    # Get the class mapping from the local files
    from classes import IMAGENET2012_CLASSES

    def iter_tar_files(tar_paths):
        for tar_path in tar_paths:
            with tarfile.open(tar_path, 'r:gz') as archive:
                for member in archive:
                    if member.name.endswith('.JPEG'):
                        # Extract image bytes
                        f = archive.extractfile(member)
                        if f is not None:
                            yield member.name, f.read()

    # Get paths to your tar files
    train_tars = [
        os.path.join(tar_dir, 'train_images_0.tar.gz'),
        os.path.join(tar_dir, 'train_images_1.tar.gz')
    ]

    images = []
    labels = []

    # Process each image
    for path, img_bytes in iter_tar_files(train_tars):
        # Extract synset ID from filename
        root, _ = os.path.splitext(path)
        _, synset_id = os.path.basename(root).rsplit("_", 1)

        images.append({'path': path, 'bytes': img_bytes})
        labels.append(IMAGENET2012_CLASSES[synset_id])

    return Dataset.from_dict({
        "image": images,
        "label": labels
    })


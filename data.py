import io
import os
import tarfile
import logging
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from transformers import CLIPImageProcessor
from datasets import load_dataset
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

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
    num_workers=0,
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
        dataset = DIYImageNet('./data', split=split, target_label=target_label)
        # dataset = DIYImagenet('/scratch4/jeisner1/imnet_files/data', split=split, target_label=target_label)

    return DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

class DIYImageNet(IterableDataset):

    def __init__(self, tar_dir, split='train', target_label=0):
        self._epoch = 0
        self._split = split
        self.tar_dir = tar_dir
        self.target_label = target_label
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(adjust_greyscale),
            transforms.Lambda(permute)
        ])
            
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        id, num_workers = worker_info.id, worker_info.num_workers
        dataset = iter_imnet(self.tar_dir, split=self._split, id=id, num_workers=num_workers)
        
        for i, item in enumerate(dataset):
            # if i % num_workers != id: continue
            label = int(item['label'])
            if label == self.target_label: continue
            img = Image.open(io.BytesIO(item['image']['bytes'])).convert('RGB')
            yield {'pixel_values': self.transform(img), 'label': torch.tensor(label, dtype=torch.long)}
            
def iter_imnet(tar_dir, split='train', id=0, num_workers=1):
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
    
    shards = tars.get(split)
    my_shards = shards[id::num_workers]
    logger.info(f'worker {id}/{num_workers} assigned shards: {my_shards}')
    
    for shard in shards[id::num_workers]:
        print(f'working on {shard}')
        with tarfile.open(shard, 'r:gz') as archive:
            for member in archive:
                if member.name.endswith('.JPEG'):
                    f = archive.extractfile(member)
                    if f is not None:
                        root, _ = os.path.splitext(member.name)
                        _, synset_id = os.path.basename(root).rsplit("_", 1)
                        yield {"image": {"path": member.name, "bytes": f.read()}, "label": to_label[synset_id]}

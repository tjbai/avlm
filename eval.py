import yaml
import logging
import argparse
import wandb
import torch
import torchvision.transforms.functional as F

from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor, MllamaForConditionalGeneration
from attack import Patch, Identity, UniversalPerturbation as Perturbation
from data import patch_loader
from classes import IMAGENET2012_CLASSES

label_to_text = {i: v for i, v in enumerate(IMAGENET2012_CLASSES.values())}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

def log_info(data, step=None):
    if wandb.run is not None: wandb.log(data, step=step)
    else: logger.info(f's{step}:{data}')


class Llava:
    def __init__(self, model='llava-hf/llava-1.5-7b-hf', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
        self.processor = AutoProcessor.from_pretrained(model)
        self.processor.patch_size = 14
        self.processor.vision_feature_select_strategy = 'default'

    def generate(self, images, prompt='What is in this image?', prefix='This image contains', max_new_tokens=32):
        prompt= f'USER: <image>\n{prompt} ASSISTANT: {prefix} '
        inputs = self.processor(images=images, text=[prompt for _ in images], return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        resp = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [r.strip() for r in resp]

class Mllama:
    def __init__(self, model='meta-llama/Llama-3.2-11B-Vision-Instruct', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = MllamaForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
        self.processor = AutoProcessor.from_pretrained(model)

    def generate(self, images, prompt='What is in this image?', prefix='This image contains', max_new_tokens=32):
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True) + prefix
        inputs = self.processor(images=images, text=[prompt for _ in images], return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
        resp = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [r.strip() for r in resp]

@torch.no_grad()
def test_attack(attack, vlm, loader, config):
    table = wandb.Table(columns=['batch', 'sample', 'label', 'label_name', 'response', 'image'])
    attack.eval()

    for i, batch in tqdm(enumerate(loader)):
        if config.get('max_steps') and i >= config['max_steps']: break

        attacked = attack.apply_attack(batch['pixel_values'].to(config['device']), normalize=False)
        pil_imgs = [F.to_pil_image(img) for img in attacked]
        resp = vlm.generate(pil_imgs, prompt=config['prompt'], prefix=config['prefix'])

        for j, (r, l, img) in enumerate(zip(resp, batch['label'], pil_imgs)):
            table.add_data(i, j, l, label_to_text.get(l.item()), r, wandb.Image(img))

    log_info({'results': table})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb', action='store_true')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(args.config)
    config['device'] = args.device

    config['wandb'] = args.wandb
    if args.wandb: wandb.init(project='avlm_eval', config=config)

    if config['attack_type'] == 'identity':
        attack = Identity()
    elif config['attack_type'] == 'patch':
        attack = Patch(model=None, target_label=None, patch_r=config['patch_r'], init_size=config['init_size'])
    elif config['attack_type'] == 'perturbation':
        attack = Perturbation(model=None, target_label=None, epsilon=config['epsilon'])
    else:
        raise NotImplementedError(f'could not match {config["attack_type"]}')

    if config.get('eval_from'):
        checkpoint = torch.load(config['eval_from'], map_location=config['device'])
        attack.load_params(checkpoint['params'])
        logger.info(f'loaded attack from {config["eval_from"]}')
    else:
        logger.info(f'did not load any attack params')

    if (family := config.get('vlm_family', 'llava')) == 'llava':
        vlm = Llava(model=config['model'])
    elif family == 'llama':
        vlm = Mllama(model=config['model'])

    # to collect accuracy, only the validation split has labels
    loader = patch_loader(split='validation', batch_size=config['batch_size'], streaming=True, target_label=config['target_label'])

    test_attack(attack, vlm, loader, config)

if __name__ == '__main__':
    main()

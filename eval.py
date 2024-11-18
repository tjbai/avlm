import yaml
import logging
import argparse
import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from attack import Patch, Identity
from data import patch_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('main')

class Llava:
    def __init__(self, model='llava-hf/llava-1.5-7b-hf', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = LlavaForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16).to(device)
        self.processor = AutoProcessor.from_pretrained(model)
        
    def generate(self, images, prompt='What is in this image?', prefix='This image contains', max_new_tokens=32):
        prefix_toks = self.processor.tokenizer(prefix, add_special_tokens=False, return_tensors='pt')['input_ids']
        decoder_input_ids = prefix_toks.repeat(len(images), 1)
        inputs = self.processor(images=images, text=[prompt for _ in images], return_tensors='pt').to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, decoder_input_ids=decoder_input_ids)
        resp = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return [r.strip() for r in resp]

@torch.no_grad()
def test_attack(attack, llava, loader, config):
    attack.eval()
    
    with open(config['output_file'], 'w') as f:
        for i, batch in tqdm(enumerate(loader)):
            if config.get('max_steps') and i >= config['max_steps']: break
            attacked = attack.apply_attack(batch['pixel_values'].to(config['device']))
            resp = llava.generate([F.to_pil_image(img) for img in attacked], prompt=config['prompt'], prefix=config['prefix'])
            for r in resp: f.write(r+'\n')
            n += len(resp)
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def main():
    args = parse_args()
    config = load_config(args.config)

    if config['attack_type'] == 'identity': attack = Identity()
    elif config['attack_type'] == 'patch': attack = Patch()
    else: raise NotImplementedError(f'could not match {config["attack_type"]}')
    
    if config.get('eval_from') :
        checkpoint = torch.load(config['eval_from'], map_location=config['device'])
        attack.load_params(checkpoint['params'])
        logger.info(f'loaded attack from {config["eval_from"]}')
    else:
        logger.info(f'did not load any trained parameters')

    llava = Llava(model=config['model'])
    loader = patch_loader(split='test', batch_size=config['batch_size'], streaming=False)

    test_attack(attack, llava, loader, config)

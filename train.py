import argparse

from transformers import CLIPForImageClassification, CLIPImageProcessor

# step 0: load llava weights into our clip
def load_from_llava():
    pass

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

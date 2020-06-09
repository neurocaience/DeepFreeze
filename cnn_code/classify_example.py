"""=============================================================================
Example script for classifying an image using a trained model.
============================================================================="""

import random
from   loadmodel import load_trained_resnet18
import os
import torch
from   torchvision import transforms
from   PIL import Image

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    images = []
    directory = 'data/CNN_context1_sorted/freeze'
    for f in os.listdir(directory):
        if f.startswith('m'):
            images.append(os.path.join(directory, f))
    directory = 'data/CNN_context1_sorted/nofreeze'
    for f in os.listdir(directory):
        if f.startswith('m'):
            images.append(os.path.join(directory, f))

    model_path = 'experiments/20190212_across/' \
                 'model-resnet18_lr-0.0001_mouse_num-0_augment-1/model.pt'
    model = load_trained_resnet18(model_path)
    model.eval()

    transform = transforms.Compose([
        transforms.CenterCrop([224, 224]),
        transforms.ToTensor()
    ])

    for _ in range(10):

        r = random.randint(0, len(images)-1)
        impath = images[r]
        target = 1 if 'nofreeze' in impath else 0
        img = Image.open(impath)
        img = transform(img).unsqueeze(0)
        img = img.float() / img.max()

        with torch.no_grad():
            output = model.forward(img)
            pred = output.argmax(dim=1).item()
            print('-' * 80)
            print(target, pred)

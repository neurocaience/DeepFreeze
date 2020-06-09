"""=============================================================================
Make classification predictions for every image in a directory
============================================================================="""

import argparse
import os
import torch

import cuda
from   data import get_image_loader
from   loadmodel import load_trained_resnet18

import time

# ------------------------------------------------------------------------------

device = cuda.device()

# ------------------------------------------------------------------------------

def main(args):
    """Classify all images in input directory.
    """
    model_path  = os.path.join(args.model_dir, args.model_name)
    model       = load_trained_resnet18(model_path)
    model       = model.to(device)

    # For classifying images in a folder: 
    # img_loader  = get_image_loader(args.in_dir, 'resnet18', 128, args.num_workers)
    # out_path    = os.path.join(args.out_file)

    # with open(out_path, 'w+') as outfile:
    #    classify_directory(model, img_loader, outfile)

    # For classifying images in folders F1a, F1b, F1c..., that belongs to root folder F1
    subFolders = os.listdir(args.in_dir)   # List the subfolders in this in_dir
    subFolders.sort()
    subFoldersPath = os.path.join(args.in_dir)
    #print(subFolders)

    for f in subFolders:
        start_time = time.time()

        subFoldersPath = os.path.join(args.in_dir, f)
        print(subFoldersPath)
        img_loader  = get_image_loader(subFoldersPath, 'resnet18', 128, args.num_workers)
        out_path    = os.path.join(args.out_dir, f +'.csv')

        with open(out_path, 'w+') as outfile:
            classify_directory(model, img_loader, outfile)
        tot_time = time.time() - start_time
        
        print(f + ', Run Time: ' + str(tot_time))

    print('done')

# ------------------------------------------------------------------------------

def classify_directory(model, img_loader, outfile):
    """Classify all images in data loader and log predictions to output file.
    """
    model.eval()

    with torch.no_grad():
        for i, (images, fpaths) in enumerate(img_loader):
            images = images.to(device)
            preds  = model.forward(images)
            inds   = preds.argmax(dim=1).tolist()
            #msg    = '\n'.join(['%s\t%s' % (fpaths[j], idx)
            #                    for j, idx in enumerate(inds)])
            msg   = '\n'.join(['%s' %(idx) 
                                for j, idx in enumerate(inds)])
            outfile.write(msg + '\n')
            print(i)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--in_dir',      type=str, required=True)
    p.add_argument('--out_dir',     type=str, required=True)  # before: out_file
    p.add_argument('--model_dir',   type=str, required=True)
    p.add_argument('--model_name',  type=str, required=True)
    args, _ = p.parse_known_args()

    args.num_workers = 4 if torch.cuda.is_available() else 0
    torch.manual_seed(args.seed)
    main(args)

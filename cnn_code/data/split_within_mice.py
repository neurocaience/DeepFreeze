"""=============================================================================
Split data into train and test sets by splitting each mouse's data. This
prevents bias in the test data that could result if one mouse was
overrepresented in the test data and behaved differently from the other mice.
============================================================================="""

import math
import numpy as np
import os
import random
import torch

# ------------------------------------------------------------------------------

BASE      = 'data/CNN_context1_sorted'
SPLIT_PCT = 0.15

# ------------------------------------------------------------------------------

def save_train_test_data(mouse2file_split, pos, neg):
    train_files  = []
    train_labels = []
    test_files   = []
    test_labels  = []

    for mouse, data in mouse2file_split.items():
        for f in data['train']:
            train_files.append(f)
            train_labels.append(get_label(f, pos, neg))
    for mouse, data in mouse2file_split.items():
        for f in data['test']:
            test_files.append(f)
            test_labels.append(get_label(f, pos, neg))

    ratio = (1 - SPLIT_PCT) / SPLIT_PCT
    assert abs(len(train_files) / len(test_files) - ratio) < 0.01
    assert abs(len(train_labels) / len(test_labels) - ratio) < 0.01

    torch.save({
        'files': train_files,
        'labels': train_labels
    }, 'data/train.pth')
    torch.save({
        'files': test_files,
        'labels': test_labels
    }, 'data/test.pth')
    print('Data saved.')

# ------------------------------------------------------------------------------

def split_files(mouse2file):
    mouse2file_split = {}

    print('Mouse\t# files\t# train\t# test\t Actual %')

    for mouse, files in mouse2file.items():
        n_samples  = len(files)
        inds       = np.array(list(range(len(files))))
        # Shuffles in-place.
        random.shuffle(inds)
        split      = math.floor(n_samples * (1 - SPLIT_PCT))
        train_inds = inds[:split]
        test_inds  = inds[split:]

        # Sanity check.
        pct = len(test_inds) / n_samples
        n_train = len(train_inds)
        n_test  = len(test_inds)
        assert n_train + n_test == n_samples
        assert abs(pct - 0.15) < 0.001
        print('%s\t%s\t%s\t%s\t%s' % (mouse, n_samples, n_train, n_test, pct))

        # Split and save.
        files       = np.array(files)
        train_files = files[train_inds]
        test_files  = files[test_inds]

        mouse2file_split[mouse] = { 'train': train_files, 'test': test_files }

    return mouse2file_split

# ------------------------------------------------------------------------------

def get_mice_to_filenames_map():
    mouse2files = {
        'm191': [],
        'm216': [],
        'm224': [],
        'm247': [],
        'm250': [],
        'm251': []
    }
    for f in os.listdir(BASE + '/freeze'):
        found = False
        for mouse in mouse2files.keys():
            if f.startswith(mouse):
                mouse2files[mouse].append(f)
                found = True
        if not found:
            print('Discarding: ' + f)
    for f in os.listdir(BASE + '/nofreeze'):
        found = False
        for mouse in mouse2files.keys():
            if f.startswith(mouse):
                mouse2files[mouse].append(f)
                found = True
        if not found:
            print('Discarding: ' + f)
    return mouse2files

# ------------------------------------------------------------------------------

def get_label(filename, pos, neg):
    if filename in pos:
        return 1
    assert filename in neg
    return 0

# ------------------------------------------------------------------------------

def main():
    print('=' * 80)
    print('Splitting data into train and test sets.')
    print('Each mouse\'s images are split into %s%% train and %s%% test.'
          % (int(100 * (1 - SPLIT_PCT)), int(100 * SPLIT_PCT)))
    print('This process is random, i.e. if you it twice, you will get two'
          'different splits.')

    pos = {f for f in os.listdir(BASE + '/nofreeze')}
    neg = {f for f in os.listdir(BASE + '/freeze')}

    # Initial sanity check.
    for f in pos.intersection(neg):
        assert not f.startswith('m')

    print('=' * 80)
    mouse2file = get_mice_to_filenames_map()
    print('=' * 80)
    mouse2file_split = split_files(mouse2file)
    print('=' * 80)
    save_train_test_data(mouse2file_split, pos, neg)
    print('=' * 80)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

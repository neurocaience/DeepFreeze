"""=============================================================================
Split data into train and test sets by selecting a single mouse as the test set.
============================================================================="""

import os
import copy
import torch
import csv                                                                     # LILI ADDED THIS

# ------------------------------------------------------------------------------

#ALL_MICE        = ['m191', 'm216', 'm224', 'm247', 'm250', 'm251']            # LILI COMMENTED OUT THIS SECTION
#MOUSE_NUM_TO_ID = {i: ALL_MICE[i] for i in range(len(ALL_MICE))}
#ROOT_DIR        = 'data/CNN_context1_sorted'

# ------------------------------------------------------------------------------

def openmIDs(mIDs_file):                                                       # LILI ADDED THIS FUNCTION
    f = open(mIDs_file)
    csv_f = csv.reader(f)
    mIDs = []
    for row in csv_f:
        mIDs.append(row[0])
    f.close()
    print('MIDs:')
    print(mIDs)
    return mIDs

def split_across(test_mouse_num,ROOT_DIR,mIDs_file):                            # LILI MODIFIED INPUTS
    """Split train and test set across a specific mouse. For example, if 
    `test_mouse_num` is `1`, this maps to mouse `m216`. The images for `m216`
    then become the test set. Everything else is the train set.
    """
    # New Code: (input test_mouse_num is a string of test mIDs separated by '_')
    test_ids = test_mouse_num.split('_')
    print('Test IDs')
    print(test_ids)

    ALL_MICE = openmIDs(mIDs_file)

    train_fname = 'data/train_%s.pth' % test_mouse_num
    test_fname  = 'data/test_%s.pth' % test_mouse_num 

    # Create set of train IDs from all mice less test mouse ID.
    train_ids = copy.deepcopy(ALL_MICE)
    for test_mouse_id in test_ids:
        train_ids.remove(test_mouse_id)
    print('Train IDs:')
    print(train_ids)

    files = get_files(ROOT_DIR + '/freeze', mIDs_file) + get_files(ROOT_DIR + '/nofreeze', mIDs_file)  # LILI MODIFIED INPUT
    train_files = get_specific_mouse_files(train_ids, files)
    test_files  = get_specific_mouse_files(test_ids, files)
    print(train_files[:10])
    print(test_files[:10])

    # Old Code (input test_mouse_num as an integer)
    # Open list of ALL_MICE                                                     # LILI ADDED
    # ALL_MICE = openmIDs(mIDs_file)

    # train_fname = 'data/train_%s.pth' % test_mouse_num
    # test_fname  = 'data/test_%s.pth' % test_mouse_num
    
    # if os.path.isfile(train_fname) and os.path.isfile(test_fname):
        # If we've already created this split before, do nothing.
    #    return

    # This is fpr convenience. It allows us to specify the mouse by an integer,
    # which in turn allow us to sweep over the range of mice.
    # MOUSE_NUM_TO_ID = {i: ALL_MICE[i] for i in range(len(ALL_MICE))}          # LILI MOVED HERE 
    # test_mouse_id = MOUSE_NUM_TO_ID[test_mouse_num]

    # Create set of train IDs from all mice less test mouse ID.
    # train_ids = copy.deepcopy(ALL_MICE)
    # del train_ids[ALL_MICE.index(test_mouse_id)]

    # files = get_files(ROOT_DIR + '/freeze', mIDs_file) + get_files(ROOT_DIR + '/nofreeze', mIDs_file)  # LILI MODIFIED INPUT
    # train_files = get_specific_mouse_files(train_ids, files)
    # test_files  = get_specific_mouse_files([test_mouse_id], files)

    assert len(train_files) + len(test_files) == len(files)

    pos = {f for f in os.listdir(ROOT_DIR + '/nofreeze')}
    neg = {f for f in os.listdir(ROOT_DIR + '/freeze')}

    train_labels = []
    for f in train_files:
        train_labels.append(get_label(f, pos, neg))

    test_labels = []
    for f in test_files:
        test_labels.append(get_label(f, pos, neg))

    # Python is synchronous, so these files will be created before training.
    torch.save({
        'files': train_files,
        'labels': train_labels
    }, train_fname)
    torch.save({
        'files': test_files,
        'labels': test_labels
    }, test_fname)

# ------------------------------------------------------------------------------

def get_files(directory, mIDs_file):                                        # LILI MODIFIED
    """Return a list of mouse files (not recursively) from a folder.
    """
    # Open list of ALL_MICE                                                     # LILI ADDED
    ALL_MICE = openmIDs(mIDs_file)

    filenames = []
    for f in os.listdir(directory):
        # Do not include extraneous files. This is not needed for correctness,
        # but allows us to verify that the number of files is what we expect.
        # if f.split('_')[0] in ALL_MICE:
        if any(mouse in f for mouse in ALL_MICE):
            filenames.append(f)
    return filenames

# ------------------------------------------------------------------------------

def get_specific_mouse_files(mouse_ids, files):
    """Return sub-list of `files` that contain an id in `mouse_ids`.
    """
    return [s for s in files if any(xs in s for xs in mouse_ids)]

# ------------------------------------------------------------------------------

def get_label(f, pos, neg):
    if f in pos:
        return 1
    assert f in neg
    return 0

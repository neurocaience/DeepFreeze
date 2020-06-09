"""=============================================================================
Load data.
============================================================================="""

from   data import ImageSet, TrainSet, TestSet
from   torch.utils.data import DataLoader
from   torch.utils.data.sampler import SubsetRandomSampler
from   torch.utils.data.sampler import SequentialSampler

# ------------------------------------------------------------------------------

def get_data_loaders(mouse_num, augment, model_name, batch_size, num_workers,
                     pin_memory, datadir):                                             #LILI ADDED datadir
    """Return data loaders for both train and test sets.
    """
    img_size = 299 if model_name == 'inception_v3' else 224
    nc       = 3 if model_name == 'inception_v3' else 1

    train_set = TrainSet(mouse_num, augment, img_size, nc, datadir)                    #LILI ADDED datadir   
    train_loader = DataLoader(
        dataset=train_set,
        sampler=SubsetRandomSampler(list(range(len(train_set)))),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )

    test_set = TestSet(mouse_num, img_size, nc, datadir)                                #LILI ADDED datadir
    test_loader = DataLoader(
        dataset=test_set,
        sampler=SubsetRandomSampler(list(range(len(test_set)))),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )

    return train_loader, test_loader

# ------------------------------------------------------------------------------

def get_image_loader(in_dir, model_name, batch_size, num_workers):
    """Return data loader for just images.
    """

    img_size = 299 if model_name == 'inception_v3' else 224

    image_set = ImageSet(in_dir, img_size)

    img_loader = DataLoader(
        dataset=image_set,
        sampler=SequentialSampler(list(range(len(image_set)))),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    return img_loader

"""=============================================================================
Train and test data sets.
============================================================================="""

from   PIL import Image, ImageFile
import os
import torch
from   torch.utils.data import Dataset
from   torchvision import transforms

# ------------------------------------------------------------------------------

class FEDataset(Dataset):

    def __init__(self, is_train, mouse_id, use_aug, img_size, nc, datadir):             # LILI MODIFIED
        if is_train:
            data_fname = 'train_%s.pth' % mouse_id
        else:
            data_fname = 'test_%s.pth' % mouse_id
        data = torch.load('data/%s' % data_fname)

        self.datadir  = datadir                                                          #LILI ADDED self.datadir
        self.use_aug  = use_aug
        self.nc       = nc
        self.files    = data['files']
        self.labels   = data['labels']
        self.augment  = transforms.Compose([
            transforms.RandomRotation((0, 360)),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop([img_size, img_size]),
            transforms.ToTensor()
        ])
        self.just_crop = transforms.Compose([
            transforms.CenterCrop([img_size, img_size]),
            transforms.ToTensor()
        ])

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        return len(self.files)

# ------------------------------------------------------------------------------

    def __getitem__(self, i):                                       
        """Return the i-th (image, label)-pair from the dataset.
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True        

        fname = self.files[i]
        label = self.labels[i]
        subdir = 'freeze' if label == 0 else 'nofreeze'
        #fpath  = 'data/CNN_context1_sorted/%s/%s' % (subdir, fname)        # LILI REMOVED
        fpath  = '%s/%s/%s' % (self.datadir, subdir, fname)                      # LILI MODIFIED

        pixels = Image.open(fpath)
        if self.use_aug:
            image = self.augment(pixels)
        else:
            # No reason to do data augmentation on the test set.
            image = self.just_crop(pixels)
        if self.nc == 3:
            image = torch.cat((image, image, image), dim=0)

        # Convert to float and normalize.
        image = image.float() / image.max()
        return image, label

# ==============================================================================

class TrainSet(FEDataset):

    def __init__(self, mouse_id, augment, img_size, nc, datadir):                                #LILI ADDED datadir
        FEDataset.__init__(self, True, mouse_id, augment, img_size, nc, datadir)                #LILI ADDED datadir

# ------------------------------------------------------------------------------

class TestSet(FEDataset):

    def __init__(self, mouse_id, img_size, nc, datadir):                                                      #LILI ADDED datadir
        FEDataset.__init__(self, False, mouse_id, False, img_size, nc, datadir)                           #LILI ADDED datadir

# ==============================================================================

class ImageSet(Dataset):
# This is for predicting dataset (not for training and testing)

    def __init__(self, in_dir, img_size):
        self.in_dir = in_dir
        self.just_crop = transforms.Compose([
            transforms.CenterCrop([img_size, img_size]),
            transforms.ToTensor()
        ])

        # Cache and index over them to ensure consistent ordering.
        files = []
        for f in os.listdir(in_dir):
            fpath = os.path.join(in_dir, f)
            if os.path.isfile(fpath) and f.endswith('.png'):
                #print(f)  # ADDED THIS LINE
                files.append(f)
        self.files = sorted(files)
        print(self.files)

# ------------------------------------------------------------------------------

    def __len__(self):
        """Return number of samples in dataset.
        """
        return len(self.files)

# ------------------------------------------------------------------------------

    def __getitem__(self, i):
        """Return the i-th (image, label)-pair from the dataset.
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        fname  = self.files[i]
        fpath  = os.path.join(self.in_dir, fname)
        pixels = Image.open(fpath)
        image = self.just_crop(pixels)
        image = image.float() / image.max()
        return image, fpath

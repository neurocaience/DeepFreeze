"""=============================================================================
Download experimental directory.
============================================================================="""

import argparse
import os

# ------------------------------------------------------------------------------

def mkdir(directory):
    """Make directory if it does not exist. Void return.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# ------------------------------------------------------------------------------

def download(directory):
    """Download directory and save locally.
    """
    remote = '/scratch/gpfs/gwg3/fe/experiments/%s' % directory
    local = '/Users/gwg/fe/experiments/'
    mkdir(local)
    cmd = 'rsync --progress -r ' \
          'gwg3@tigergpu.princeton.edu:%s %s' % (remote, local)
    os.system(cmd)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=str, required=True)
    args = p.parse_args()
    download(args.directory)

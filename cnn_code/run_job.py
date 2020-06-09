"""=============================================================================
Script for training models on Tiger GPU.
============================================================================="""

import argparse
from   copy import copy
import datetime
import itertools
import os
import sys

# ------------------------------------------------------------------------------

def main(args):
    """Run jobs for entire input parameter space.
    """
    iterables   = []
    iter_fields = []
    for f in vars(args):
        a = getattr(args, f)
        if type(a) is list:
            iterables.append(a)
            iter_fields.append(f)

    for a in itertools.product(*iterables):
        for f in iter_fields:
            setattr(args, f, None)
        cargs = copy(args)
        for i, f in enumerate(iter_fields):
            setattr(cargs, f, a[i])

        cargs.directory  = gen_subdir(cargs, iter_fields)

        mem    = cargs.mem
        args_dict = vars(cargs)
        del args_dict['mem']
        del args_dict['root']
        del args_dict['exec_cmd']

        contents = gen_sbatch_file(args.exec_cmd, args_dict, mem=mem,
                                   wall_time=cargs.wall_time)
        run_job(cargs.directory, contents)

# ------------------------------------------------------------------------------

def run_job(fpath, sbatch_contents):
    """Create sbatch file and run job.
    """
    sbatch_fname = '%s/sbatch.sh' % fpath
    with open(sbatch_fname, 'w+') as f:
        f.write(sbatch_contents)
    cmd = 'sbatch %s' % sbatch_fname
    os.system(cmd)

# ------------------------------------------------------------------------------

def gen_subdir(cargs, iter_fields):
    """Return subdirectory name based on experimental setup.
    """
    # Create root directory.
    mkdir(cargs.root)
    desc = []
    for f in iter_fields:
        desc.append('%s-%s' % (f, getattr(cargs, f)))
    subdir = cargs.root + '/' + '_'.join(desc)
    mkdir(subdir)
    return subdir

# ------------------------------------------------------------------------------

def mkdir(directory):
    """Make directory if necessary.
    """
    if not os.path.isdir(directory):
        os.system('mkdir %s' % directory)

# ------------------------------------------------------------------------------

def gen_sbatch_file(run_job_cmd, args, mem, wall_time):
    """Return contents of sbatch file based on experimental setup.
    """
    n_gpus = 1
    logfile = '%s/out.txt' % args['directory']
    if type(mem) is int:
        mem = str(mem) + 'G'
    # The -O flag means "optimized". Also disables assert statements.
    script_cmd = 'python -O train.py %s' % args_to_cmds(args)

    header = """#!/bin/bash

# The Python script and arguments used to generate this file:
#
# python %s

#SBATCH --mail-user=lxcai@princeton.edu
#SBATCH --mail-type=end
#SBATCH --mem %s
#SBATCH --nodes=1
#SBATCH --gres=gpu:tesla_p100:%s
#SBATCH --ntasks-per-node=5
#SBATCH --output=%s
#SBATCH --time=%s:00:00

#Greg's Environment
#module load cudatoolkit/8.0 cudann/cuda-8.0/5.1
#module load anaconda3
#source activate fe
#cd /scratch/gpfs/gwg3/fe

#Lili's Environment
module load anaconda3/5.3.1
source activate ptorch
#cd /tigress/lxcai/fe.git/trunk\n
cd /jukebox/witten/Lili/ManualScoring/CNN_analysis/git/fe\n

""" % (run_job_cmd, mem, n_gpus, logfile, wall_time)
    return header + script_cmd

# ------------------------------------------------------------------------------

def args_to_cmds(args):
    """Convert args dictionary to command line arguments. For example:

    >>> args_to_cmds({ 'foo': 'bar', 'qux': 'baz' })
    '--foo=bar --qux=baz'
    """
    result = ''
    for key, val in args.items():
        result += '--%s=%s ' % (key, val)
    return result

# ------------------------------------------------------------------------------

def get_now_str():
    """Return date string for experiments folders, e.g.: '20180621'.
    """
    now     = datetime.datetime.now()
    day     = '0%s' % now.day if now.day < 10 else now.day
    month   = '0%s' % now.month if now.month < 10 else now.month
    now_str = '%s%s%s' % (now.year, month, day)
    return now_str

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    now_str = get_now_str()
    p = argparse.ArgumentParser()

    #p.add_argument('--root',
    #               required=True,
    #               type=lambda s: 'experiments/%s_%s' % (now_str, s))
    p.add_argument('--root',       type=str,   default='/tigress/lxcai/experiments')
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--mem',        type=int,   default=150)
    p.add_argument('--model',      type=str,   default=['resnet18'],
                   choices=['alexnet', 'inception_v3', 'resnet18'], nargs='*')
    p.add_argument('--lr',         type=float, default=[1e-3], nargs='*')
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--n_epochs',   type=int,   default=200)
    p.add_argument('--mouse_num',  type=str,   default='m216',    nargs='*')
    p.add_argument('--augment',    type=int,   default=[1],    nargs='*')
    p.add_argument('--pretrained', type=int,   default=[0],    nargs='*')
    p.add_argument('--datadir',    type=str,   default='data/CNN_context1_sorted')  # Lili added
    p.add_argument('--mIDs',       type=str,   default='data/mIDs.csv')             # Lili added
    p.add_argument('--name',       type=str,   default='EXTgc')                     # Lili added
    
    args = p.parse_args()

    # Cache the command used to run this script. This way experiments are more
    # reproducible.
    args.exec_cmd = " ".join(sys.argv[:])

    main(args)

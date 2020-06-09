"""=============================================================================
Train a convolutional neural network on fear extinction data.
============================================================================="""

import argparse
import time
import os

import torch
import torch.utils.data
from   torch import optim
from   torch import nn

from torchvision.models import inception_v3, resnet18

import cuda
from   data import get_data_loaders, split_across

# ------------------------------------------------------------------------------

SAVE_MODEL_EVERY = 5
device = cuda.device()

# ------------------------------------------------------------------------------

def main(args):
    """Train fear extinction CNN.
    """
    start_time = time.time()
    print('-' * 80)
    log_args(args)
    print('-' * 80)

    model = load_model(args.model, args.pretrained)
    model = model.to(device)
    print('Model loaded.')
    
    split_across(args.mouse_num, args.datadir, args.mIDs)
    print('Train / test sets created (if necessary).')

    train_loader, test_loader = get_data_loaders(args.mouse_num,
                                                 args.augment,
                                                 args.model,
                                                 args.batch_size,
                                                 args.num_workers,
                                                 args.pin_memory,
                                                 args.datadir)                 
    print('Data loaded.')

    print('Training.')
    print('-' * 80)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.n_epochs + 1):
        train_loss = train(model, criterion, optimizer, train_loader)
        test_loss, pct_correct, f0, f1, t0, t1 = test(model, criterion, test_loader)

        out_file = args.name + '_pretrained' + str(int(args.pretrained==True)) + '.csv'
        out_path = os.path.join(args.directory, out_file)
        
        msg = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (epoch, round(train_loss,2), round(test_loss,2), pct_correct,f0,f1,t0,t1)
        print(msg)
        
        msg2 = ','.join([str(epoch), str(round(train_loss,2)), str(round(test_loss,2)),
                        str(pct_correct), str(f0), str(f1), str(t0), str(t1)]) #round(test_loss,2), pct_correct, f0, f1, t0, t1])
        with open(out_path, 'a+') as outfile:
            outfile.write(msg2 + '\n') 

        if epoch % SAVE_MODEL_EVERY == 0: 
            name = args.name + '_epoch%s' %(str(epoch))
            save_model(args.directory, model, name) 

    hours = round((time.time() - start_time) / 3600, 1)
    print('Job complete in %s hrs.' % hours)

    save_model(args.directory, model, args.name)
    print('Model saved.')

# ------------------------------------------------------------------------------

def train(model, criterion, optimizer, train_loader):
    """Perform stochastic SGD on train set.
    """
    model.train()
    loss_sum = 0


    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        image = image.to(device)
        label = label.to(device)
        pred  = model.forward(image)
        loss  = criterion(pred, label.long())
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / i

# ------------------------------------------------------------------------------

def test(model, criterion, test_loader):
    """Perform stochastic SGD on test set.
    """
    model.eval()
    loss_sum = 0
    n_samples = 0
    n_correct = 0
    n_false0 = 0                                                            
    n_false1 = 0                                                            
    n_true0 = 0
    n_true1 = 0

    with torch.no_grad():
       
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device)
            label = label.to(device)
            pred  = model.forward(image)

            # Compute accuracy of batch.
            inds = pred.argmax(dim=1)
            n_correct += (label == inds).sum().item()
            n_false0 += (label - inds == 1).sum().item()                    
            n_false1 += (label - inds == -1).sum().item()                   
            n_true0 += (label + inds == 0).sum().item()
            n_true1 += (label + inds == 2).sum().item() 
            n_samples += len(label)

            loss = criterion(pred, label.long())
            loss_sum += loss.item()

    pct_correct = round(100 * (n_correct / n_samples), 2)
    loss_avg = loss_sum / i

    return loss_avg, pct_correct, n_false0, n_false1, n_true0, n_true1

# ------------------------------------------------------------------------------

def load_model(model_name, pretrained):
    """Return appropriate architecture based on model name string.
    """
    if model_name == 'resnet18':
        model = resnet18(pretrained=pretrained)
        # Manually fix resnet18 to handle black and white images.
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        # Manually set the number of classes to 2. We can't initialize to 2 in
        # case of pretraining.
        model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    elif model_name == 'inception_v3':
        model = inception_v3(pretrained=pretrained)#, transform_input=False)
        # Manually fix inception_v3 to handle black and white images.
        # model.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        # Set this to false so models have a unified output. See:
        # https://github.com/pytorch/vision/issues/302#issuecomment-341163548
        model.aux_logits = False
        # Manually set the number of classes to 2. We can't initialize to 2 in
        # case of pretraining.
        model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
    elif model_name == 'alexnet':
        raise NotImplemented('AlexNet not supported yet.')
    return model

# ------------------------------------------------------------------------------

def save_model(directory, model, name):
    """Save PyTorch model's state dictionary for provenance.
    """
    fpath = '%s/model-%s.pt' % (directory, name)
    state = model.state_dict()
    torch.save(state, fpath)

# ------------------------------------------------------------------------------

def log_args(args):
    """Print arguments passed to script.
    """
    fields = [f for f in vars(args)]
    longest = max(fields, key=len)
    format_str = '{:>%s}  {:}' % len(longest)
    for f in fields:
        msg = format_str.format(f, getattr(args, f))
        print(msg)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--directory',  type=str,   default='experiments/local')
    p.add_argument('--wall_time',  type=int,   default=24)
    p.add_argument('--seed',       type=int,   default=0)
    p.add_argument('--model',      type=str,   default='resnet18',
                   choices=['alexnet', 'inception_v3', 'resnet18'])
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--n_epochs',   type=int,   default=200)
    p.add_argument('--mouse_num',  type=str,   default='m216')
    p.add_argument('--augment',    type=int,   default=1)
    p.add_argument('--pretrained', type=int,   default=0)
    p.add_argument('--datadir',    type=str,   default='data/CNN_context1_sorted')  # Lili added
    p.add_argument('--mIDs',       type=str,   default='data/mIDs.csv')             # Lili added
    p.add_argument('--name',       type=str,   default='EXTgc')                     # Lili added

    args, _ = p.parse_known_args()

    args.augment     = bool(args.augment)
    args.pretrained  = bool(args.pretrained)
    args.pin_memory  = torch.cuda.is_available()
    args.num_workers = 0 if args.directory == 'experiments/local' else 4

    torch.manual_seed(args.seed)
    main(args)

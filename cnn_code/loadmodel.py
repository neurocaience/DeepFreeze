"""=============================================================================
Utility script for loading PyTorch models from state dictionaries.

Example
-------
Start REPL in root directory:
>>> from loadmodel import load_trained_resnet18
>>> model = load_trained_resnet18('/path/to/model.pt')
============================================================================="""

import torch
from   torch import nn
from   torchvision.models import inception_v3, resnet18
from   torchvision.models.inception import BasicConv2d

# ------------------------------------------------------------------------------

def load_trained_resnet18(fname):
    """Load trained resnet18 from path to its state dictionary.
    """
    model = resnet18(pretrained=False, num_classes=2)
    conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1 = conv1
    return _load_model_from_state_dict(model, fname)

# ------------------------------------------------------------------------------

def load_trained_inception_v3(fname):
    """Load trained inception_v3 from path to its state dictionary.
    """
    model = inception_v3(pretrained=False, num_classes=2)
    model.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
    return _load_model_from_state_dict(model, fname)

# ------------------------------------------------------------------------------

def _load_model_from_state_dict(model, fname):
    """Load trained model based on instance and path to state dictionary.
    """
    state_dict = torch.load(fname, map_location={'cuda:0': 'cpu'})
    model.load_state_dict(state_dict)
    return model

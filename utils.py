import random, torch, os
import numpy as np
import torch.nn as nn
import copy
import config


def save_checkpoint(model, optimizer, filename='my_checkpoint.pth.tar'):
    print('=> sabing checkpoint')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model: nn.Module, optimizer: torch.optim.Optimizer, lr):
    print('=> loading checkpoint')

    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

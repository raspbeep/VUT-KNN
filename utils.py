import os

import torch
import torch.nn as nn

import config

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_to_checkpoint(epoch, gen_c1, gen_c2, opt_gen, disc_c1, disc_c2, opt_disc, filename=config.CHECKPOINT_ALL):
    print(f'=> saving checkpoint at epoch {epoch}')
    checkpoint = {
        'epoch': epoch,
        'generators': {
            'gen_c1': gen_c1.state_dict(),
            'gen_c2': gen_c2.state_dict(),
            'opt_gen': opt_gen.state_dict(),
        },
        'discriminators' : {
            'disc_c1': disc_c1.state_dict(),
            'disc_c2': disc_c2.state_dict(),
            'opt_disc': opt_disc.state_dict(),
        }   
    }
    torch.save(checkpoint, filename)

def load_from_checkpoint(gen_c1, gen_c2, opt_gen, disc_c1, disc_c2, opt_disc, lr, filename=config.CHECKPOINT_ALL):
    if not os.path.isfile(filename):
        print(f"No checkpoint found at '{filename}'")
        return None
    
    checkpoint = torch.load(filename, map_location=config.DEVICE)
    checkpoint = torch.load(filename)

    epoch = checkpoint['epoch']

    gen_c1.load_state_dict(checkpoint['generators']['gen_c1'])
    gen_c2.load_state_dict(checkpoint['generators']['gen_c2'])
    opt_gen.load_state_dict(checkpoint['generators']['opt_gen'])

    disc_c1.load_state_dict(checkpoint['discriminators']['disc_c1'])
    disc_c2.load_state_dict(checkpoint['discriminators']['disc_c2'])
    opt_disc.load_state_dict(checkpoint['discriminators']['opt_disc'])

    for param_group in opt_gen.param_groups:
        param_group["lr"] = lr

    for param_group in opt_disc.param_groups:
        param_group["lr"] = lr

    return epoch

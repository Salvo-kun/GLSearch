from train import train
from test import test
from options import opt
import os

"""
This file is used for debug purposes, to be able to specify our options from a script (options.debug_options) instead of the command line.
"""

if __name__ == '__main__':
    if opt.phase == 'train':
        train()
    elif opt.phase == 'test':
        test()
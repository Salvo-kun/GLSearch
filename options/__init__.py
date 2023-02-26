import os
import sys

# get name of caller (test, train or debug)
caller = sys.argv[0].split(os.sep)[-1].split(".")[0]


# depending on caller, import the correct options
opt = None
if caller == "train":
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
elif caller == "test":
    from options.test_options import TestOptions
    opt = TestOptions().parse()
elif caller == "debug":
    from options.debug_options import DebugOptions
    opt = DebugOptions()

if opt is None:
    raise ValueError("opt is None. you should call this script from train.py, test.py or debug.py.")
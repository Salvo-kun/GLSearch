import os
from .base_options import BaseOptions
import copy

"""Debug options for the project."""

class DebugOptionsObject:
    def __init__(self, opt):
        # copy all the options from the base options
        for k, v in opt.__dict__.items():
            self.__dict__[k] = copy.deepcopy(v)

        # set the new/overrided options
        self.log_level = 'debug'
        self.log_file = None
        self.log_stdout = True

        self.phase = 'train'
        self.device = 'cpu'
        self.data_folder = os.path.join("data", "dataset_files")
        if True:
            self.dataset_list = [
            ([('aids700nef', 30),
              ('linux', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=16,ed=5,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=14,ed=0.14,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=18,ed=0.2|2,gen_type=WS', -1),
              ], 2500),
            ([('ptc', 30),
              #('imdbmulti', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=32,ed=4,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=30,ed=0.12,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=34,ed=0.2|2,gen_type=WS', -1),
              ], 2500),
            ([('mutag', 30),
              ('redditmulti10k', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=48,ed=4,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=46,ed=0.1,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=50,ed=0.2|4,gen_type=WS', -1),
              ], 2500),
            ([('webeasy', 30),
              ('mcsplain-connected', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=64,ed=3,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=62,ed=0.08,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=66,ed=0.2|4,gen_type=WS', -1),
              ], 2500)
        ]
        else:
            self.dataset_list = [
                ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
            ]
        self.shuffle_input = False
        self.batch_size = 1
        # feature encoders
        self.node_fe_1 = 'one_hot'
        self.node_fe_2 = 'local_degree_profile'

class DebugOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

    def parse(self):
        super().parse()
        return DebugOptionsObject(self.opt)





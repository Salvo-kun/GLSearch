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
        self.export_computed_pairs = False

        self.phase = 'train'
        self.device = 'cpu'
        self.data_folder = os.path.join("data", "dataset_files")
        if False:
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
        elif False:
            self.dataset_list = [
                ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
            ]
        elif True:
            self.dataset_list = [
                ([('ptc', 30)], 2500),
            ]
        else:
            self.dataset_list = [
                ([#('webeasy', 30),
                  ('mcsplain-connected', 30),
                  ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=64,ed=3,gen_type=BA', -1),
                  ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=62,ed=0.08,gen_type=ER', -1),
                  ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=66,ed=0.2|4,gen_type=WS', -1),
                  ], 2500),
            ]
        self.plot_graphs_on_loading = True

        # FIXME legacy
        self.scalable = True

        # TODO check which options are used
        self.shuffle_input = False
        self.model = 'MCSRL_backtrack'
        self.dataset = 'cocktail'
        # TODO write better the layer_1 config
        # original: self.layer_1 = 'MCSRL_backtrack:Q_sampling=canonical_0.000016_0.16_0.01,DQN_mode=tgt_q_network,Q_BD=True,loss_fun=mse,q_signal=fitted-tgt-random-path,sync_target_frames=100,beta_reward=0,perc_IL=-1,buffer_start_iter=11,buffer_size=1024,sample_size=32,sample_all_edges=False,sample_all_edges_thresh=-1,eps_testing=False,recursion_threshold=-1,total_runtime=-1,save_every_recursion_count=-1,save_every_runtime=-1,mcsplit_heuristic_on_iter_one=False,restore_bidomains=False,no_pruning=False,regret_iters=3,populate_reply_buffer_every_iter=-1,encoder_type=abcd,embedder_type=abcd,interact_type=dvn,n_dim=64,n_layers=3,GNN_mode=GAT,learn_embs=True,layer_AGG_w_MLP=True,Q_mode=8,Q_act=elu+1,disentangle_search_tree=False,mcsp_before_perc=0.1'
        self.layer_1 = 'MCSRL_backtrack:Q_sampling=canonical_0.000016_0.16_0.01,DQN_mode=tgt_q_network,Q_BD=True,loss_fun=mse,q_signal=fitted-tgt-random-path,sync_target_frames=100,beta_reward=0,perc_IL=-1,buffer_start_iter=11,buffer_size=1024,sample_size=32,sample_all_edges=False,sample_all_edges_thresh=-1,eps_testing=False,recursion_threshold=-1,total_runtime=-1,save_every_recursion_count=-1,save_every_runtime=-1,mcsplit_heuristic_on_iter_one=False,restore_bidomains=False,no_pruning=False,regret_iters=3,populate_reply_buffer_every_iter=-1,encoder_type=abcd,embedder_type=abcd,n_dim=64,n_layers=3,GNN_mode=GAT,learn_embs=True,layer_AGG_w_MLP=True,Q_mode=8,Q_act=elu+1,disentangle_search_tree=False,mcsp_before_perc=0.1'
        self.layer_num = 1

        self.loss_func = 'MSE'
        self.smarter_bin_sampling = False
        self.smart_bin_sampling = True
        self.no_trivial_pairs = True
        self.reward_calculator_mode = 'vanilla'
        self.interact_ops = ['32', '1dconv+max_1', 'add']
        self.emb_mode_list = ['gs', 'sgs', 'abds', 'ubds']
        self.num_nodes_dqn_max = -1
        self.simplified_sg_emb = True
        self.run_bds_MLP_before_interact = False
        self.default_emb = 'learnable'
        self.with_bdgnn = False
        self.with_gnn_per_action = False
        self.total_runtime = -1
        self.recursion_threshold = 7500
        self.time_analysis = False
        self.val_every_iter = 100 if self.phase == 'train' else 1
        self.supervised_before = 1
        self.imitation_before = 1
        self.a2c_networks = False
        self.val_debug = False
        self.batch_size = 1
        self.num_bds_max = 1
        self.num_nodes_degree_max = 20
        self.num_nodes_dqn_max = -1
        self.val_method_list = ['dqn']
        self.logging = 'end'
        self.inverse_bd_size_order = False
        self.use_mcsp_policy = False
        self.exclude_root = False
        self.plot_final_tree = True
        self.randQ = False
        if self.phase == 'train':
            self.node_feats_for_mcs = []
        else:
            # TODO in test phase, original might have different values
            self.node_feats_for_mcs = ['type']

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






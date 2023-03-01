import argparse
import os
from utils.validation import null_coalescence

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        dataset_list = [
            ([('aids700nef', 30),
              ('linux', 30),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=16,ed=5,gen_type=BA', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=14,ed=0.14,gen_type=ER', -1),
              ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=18,ed=0.2|2,gen_type=WS', -1),
              ], 2500),
            # ([('ptc', 30),
            #   # ('imdbmulti', 30),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=32,ed=4,gen_type=BA', -1),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=30,ed=0.12,gen_type=ER', -1),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=34,ed=0.2|2,gen_type=WS', -1),
            #   ], 2500),
            # ([('mutag', 30),
            #   ('redditmulti10k', 30),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=48,ed=4,gen_type=BA', -1),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=46,ed=0.1,gen_type=ER', -1),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=50,ed=0.2|4,gen_type=WS', -1),
            #   ], 2500),
            # ([('webeasy', 30),
            #   # ('mcsplain-connected', 30),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=64,ed=3,gen_type=BA', -1),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=62,ed=0.08,gen_type=ER', -1),
            #   ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=66,ed=0.2|4,gen_type=WS', -1),
            #   ], 2500)
        ]
        
        layer_1 = 'MCSRL_backtrack:Q_sampling=canonical_0.000016_0.16_0.01,DQN_mode=tgt_q_network,Q_BD=True,loss_fun=mse,q_signal=fitted-tgt-random-path,sync_target_frames=100,beta_reward=0,perc_IL=-1,buffer_start_iter=11,buffer_size=1024,sample_size=32,sample_all_edges=False,sample_all_edges_thresh=-1,eps_testing=False,recursion_threshold=-1,total_runtime=-1,save_every_recursion_count=-1,save_every_runtime=-1,mcsplit_heuristic_on_iter_one=False,restore_bidomains=False,no_pruning=False,regret_iters=3,populate_reply_buffer_every_iter=-1,encoder_type=abcd,embedder_type=abcd,interact_type=dvn,n_dim=64,n_layers=3,GNN_mode=GAT,learn_embs=True,layer_AGG_w_MLP=True,Q_mode=8,Q_act=elu+1,disentangle_search_tree=False,mcsp_before_perc=0.1' 
        
        self.parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Specify the device to use. Can be "cpu" or "cuda".')
        self.parser.add_argument('--data_folder', type=str, default=os.path.join("data", "dataset_files"), help='path to the folder containing the dataset files (in pickle json format)')
        self.parser.add_argument('--dataset_list', type=list, default=dataset_list, help='list of datasets to train on')
        self.parser.add_argument('--shuffle_input', type=bool, default=False, help='')
        self.parser.add_argument('--batch_size', type=int, default=1, help='')        
        self.parser.add_argument('--model', default='MCSRL_backtrack')
        self.parser.add_argument('--dataset', default='cocktail')
        self.parser.add_argument('--load_model', default=None)
        self.parser.add_argument('--split_by', default='graph')
        self.parser.add_argument('--dataset_version', default=None)
        self.parser.add_argument('--filter_large_size', type=int, default=None)
        self.parser.add_argument('--select_node_pair', type=str, default=None)
        self.parser.add_argument('--model_name', default='fancy')
        self.parser.add_argument('--plot_final_tree', type=bool, default=True)
        self.parser.add_argument('--no_trivial_pairs', type=bool, default=True)
        self.parser.add_argument('--no_bd_MLPs', type=bool, default=False)
        self.parser.add_argument('--smarter_bin_sampling', type=bool, default=False)
        self.parser.add_argument('--smart_bin_sampling', type=bool, default=True)
        self.parser.add_argument('--logging', type=str, default='end')
        self.parser.add_argument('--search_path', type=bool, default=True)
        self.parser.add_argument('--with_bdgnn', type=bool, default=False)
        self.parser.add_argument('--with_gnn_per_action', type=bool, default=False)
        self.parser.add_argument('--max_chunk_size', type=int, default=64)
        self.parser.add_argument('--a2c_networks', type=bool, default=False)
        self.parser.add_argument('--interact_ops', default=['32', '1dconv+max_1', 'add'])
        self.parser.add_argument('--run_bds_MLP_before_interact', type=bool, default=False)
        self.parser.add_argument('--inverse_bd_size_order', type=bool, default=False)
        self.parser.add_argument('--num_bds_max', type=int, default=1)
        self.parser.add_argument('--num_nodes_dqn_max', type=int, default=-1)
        self.parser.add_argument('--val_every_iter', type=int, default=None) 
        self.parser.add_argument('--val_debug', type=bool, default=False)
        self.parser.add_argument('--clipping_val', type=float, default=-1)
        self.parser.add_argument('--loss_func', default='MSE')
        self.parser.add_argument('--supervised_before', type=int, default=None)
        self.parser.add_argument('--imitation_before', type=int, default=None)
        self.parser.add_argument('--attention_bds', type=bool, default=False)
        self.parser.add_argument('--simplified_sg_emb', type=bool, default=True)
        self.parser.add_argument('--emb_mode_list', type=list, default=['gs', 'sgs', 'abds', 'ubds'])
        self.parser.add_argument('--default_emb', default='learnable')
        self.parser.add_argument('--normalize_emb', type=bool, default=True)
        self.parser.add_argument('--batched_logging', default=True)
        self.parser.add_argument('--randQ', default=False)
        self.parser.add_argument('--val_method_list', default=['dqn'])#,'mcspv2','mcsprl'])
        self.parser.add_argument('--use_mcsp_policy', type=bool, default=False)
        self.parser.add_argument('--layer_num', type=int, default=1) 
        self.parser.add_argument('--layer_1', type=str, default=layer_1)
        self.parser.add_argument('--recursion_threshold', type=int, default=None)
        self.parser.add_argument('--total_runtime', type=int, default=None)
        self.parser.add_argument('--num_nodes_degree_max', type=int, default=None)
        self.parser.add_argument('--reward_calculator_mode', default='vanilla')
        self.parser.add_argument('--node_feats_for_mcs', default=['type']) 
        self.parser.add_argument('--tvt_strategy', default='holdout')
        self.parser.add_argument('--train_test_ratio', type=float, default=0.8)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--retain_graph', type=bool, default=None) 
        self.parser.add_argument('--periodic_save', type=int, default=100)
        self.parser.add_argument('--validation', type=bool, default=False)
        self.parser.add_argument('--throw_away', type=float, default=0)
        self.parser.add_argument('--print_every_iters', type=int, default=5)
        self.parser.add_argument('--only_iters_for_debug', type=int, default=None)
        self.parser.add_argument('--time_analysis', type=bool, default=False)
        self.parser.add_argument('--save_model', type=bool, default=True)
        self.parser.add_argument('--node_ordering', default=None) 
        self.parser.add_argument('--scalable', type=bool, default=None) 
        self.parser.add_argument('--exclude_root', default=False)
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
            
        self.opt = self.parser.parse_args()
        
        if self.opt.load_model:
            self.opt.supervised_before = null_coalescence(self.opt.supervised_before, -1) 
            self.opt.imitation_before = null_coalescence(self.opt.imitation_before, -1) 
            self.opt.val_every_iter = null_coalescence(self.opt.val_every_iter, 1)
            self.opt.retain_graph = null_coalescence(self.opt.retain_graph, False)
        else:
            self.opt.supervised_before = null_coalescence(self.opt.supervised_before, 1250)
            self.opt.imitation_before = null_coalescence(self.opt.imitation_before, 3750)
            self.opt.val_every_iter = null_coalescence(self.opt.val_every_iter, 100)
            self.opt.retain_graph = null_coalescence(self.opt.retain_graph, True)
            self.opt.recursion_threshold = null_coalescence(self.opt.recursion_threshold, 80)            
            
        self.opt.node_ordering = null_coalescence(self.opt.node_ordering, None if 'syn' in self.opt.dataset or 'pdb' in self.opt.dataset else 'bfs')
        self.opt.total_runtime = null_coalescence(self.opt.total_runtime, -1)
        self.opt.scalable = null_coalescence(self.opt.scalable, True if len(self.opt.dataset_list) == 1 else False)
                
        return self.opt
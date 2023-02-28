import os

"""Debug options for the project."""

class DebugOptions:
    def __init__(self):
        self.phase = 'train'
        self.device = 'cpu'
        self.data_folder = os.path.join("data","dataset_files")
        self.dataset_list = [
                   ([('duogexfroadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2', 1)], 1),
        ]
        self.model = 'MCSRL_backtrack'
        self.dataset = 'cocktail'
        self.layer_1 = 'MCSRL_backtrack:Q_sampling=canonical_0.000016_0.16_0.01,DQN_mode=tgt_q_network,Q_BD=True,loss_fun=mse,q_signal=fitted-tgt-random-path,sync_target_frames=100,beta_reward=0,perc_IL=-1,buffer_start_iter=11,buffer_size=1024,sample_size=32,sample_all_edges=False,sample_all_edges_thresh=-1,eps_testing=False,recursion_threshold=-1,total_runtime=-1,save_every_recursion_count=-1,save_every_runtime=-1,mcsplit_heuristic_on_iter_one=False,restore_bidomains=False,no_pruning=False,regret_iters=3,populate_reply_buffer_every_iter=-1,encoder_type=abcd,embedder_type=abcd,interact_type=dvn,n_dim=64,n_layers=3,GNN_mode=GAT,learn_embs=True,layer_AGG_w_MLP=True,Q_mode=8,Q_act=elu+1,disentangle_search_tree=False,mcsp_before_perc=0.1' 
        self.layer_num = 1
        self.scalable = 'False'        
        self.loss_func = 'MSE'
        self.smarter_bin_sampling = 'False'
        self.smart_bin_sampling = 'True'
        self.no_trivial_pairs = 'True'
        self.reward_calculator_mode = 'vanilla'
        self.interact_ops = ['32', '1dconv+max_1', 'add']
        self.emb_mode_list = ['gs', 'sgs', 'abds', 'ubds']
        self.num_nodes_dqn_max = -1
        self.simplified_sg_emb = 'True'
        self.run_bds_MLP_before_interact = 'False'
        self.default_emb = 'learnable'
        self.with_bdgnn = 'False'
        self.with_gnn_per_action = 'False'
        self.total_runtime = -1
        self.recursion_threshold = 7500
        self.time_analysis = 'False'
        self.val_every_iter = 1
        self.supervised_before = 1
        self.imitation_before = 1
        self.a2c_networks = 'False'
        self.val_debug = 'False'
        self.batch_size = 1
        self.num_bds_max = 1
        self.num_nodes_degree_max = 20
        self.num_nodes_dqn_max = -1
        self.val_method_list = ['dqn']
        self.logging = 'end'
        self.inverse_bd_size_order = 'False'
        self.use_mcsp_policy = 'False'
        self.exclude_root = 'False'
        self.plot_final_tree = 'True'
        self.randQ = ' False'
        self.node_feats_for_mcs = ['type']

from models.gl_search import GLSearch

if __name__ == '__main__':
    opt = {}
    
    opt['layer_1'] = 'MCSRL_backtrack:Q_sampling=canonical_0.000016_0.16_0.01,DQN_mode=tgt_q_network,Q_BD=True,loss_fun=mse,q_signal=fitted-tgt-random-path,sync_target_frames=100,beta_reward=0,perc_IL=-1,buffer_start_iter=11,buffer_size=1024,sample_size=32,sample_all_edges=False,sample_all_edges_thresh=-1,eps_testing=False,recursion_threshold=-1,total_runtime=-1,save_every_recursion_count=-1,save_every_runtime=-1,mcsplit_heuristic_on_iter_one=False,restore_bidomains=False,no_pruning=False,regret_iters=3,populate_reply_buffer_every_iter=-1,encoder_type=abcd,embedder_type=abcd,interact_type=dvn,n_dim=64,n_layers=3,GNN_mode=GAT,learn_embs=True,layer_AGG_w_MLP=True,Q_mode=8,Q_act=elu+1,disentangle_search_tree=False,mcsp_before_perc=0.1' 
    opt['layer_num'] = 1
    
    model = GLSearch(opt)
    print(model)
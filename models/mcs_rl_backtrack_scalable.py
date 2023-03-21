from __future__ import annotations
from collections import defaultdict
import random
from torch.nn import MSELoss, BCEWithLogitsLoss
from models.base_model import BaseModel
from utils.options import parse_bool
from utils.validation import validate, IS_POSITIVE, IS_BETWEEN_0_AND_1
from copy import deepcopy
import numpy as np
import torch
from torch.nn.functional import softmax
from utils.graph import create_edge_index, create_adj_set
from utils.timer import OurTimer
from utils.data_structures.search_tree import Bidomain, StateNode, ActionEdge, SearchTree, ActionSpaceDataScalable, \
    unroll_bidomains, get_natts_hash, get_natts2g2abd_sg_nids
from utils.data_structures.buffer import BinBuffer
from utils.data_structures.common import StackHeap, DoubleDict
from utils.data_structures.dqn_input import DQNInput
from models.dqn import Q_network_v1
from utils.reward_calculator import RewardCalculator
from utils.saver import saver
from options import opt
from utils.mc_split import McspVec, BufferEntry, ForwardConfig, MethodConfig, IMITATION_MODE, PRETRAIN_MODE, TRAIN_MODE, \
    TEST_MODE


#########################################################################
# MCSRL Procedure
#########################################################################

class MCSplitRLBacktrackScalable(BaseModel):
    def __init__(self, **kwargs):
        super(MCSplitRLBacktrackScalable, self).__init__()

        self.DQN_mode = validate(kwargs['DQN_mode'], str)
        self.q_signal = validate(kwargs['q_signal'], str)
        self.recursion_threshold = validate(int(kwargs['recursion_threshold']), int, IS_POSITIVE, None)
        self.total_runtime = validate(int(kwargs['total_runtime']), int, IS_POSITIVE, None)
        self.restore_bidomains = validate(parse_bool(kwargs['restore_bidomains']), bool)
        self.regret_iters = validate(int(kwargs['regret_iters']), int, IS_POSITIVE, None)
        self.buffer_size = validate(int(kwargs['buffer_size']), int, IS_POSITIVE)
        self.global_loss_func = validate(opt.loss_func, str,
                                         lambda x: x in ['MSE', 'BCEWithLogits'])  # TODO validate inside config
        self.Q_sampling = validate(kwargs['Q_sampling'], str)
        self.feat_map = validate(kwargs['feat_map'], dict)
        self.sync_target_frames = validate(int(kwargs['sync_target_frames']), int, IS_POSITIVE)
        self.buffer_start_iter = validate(int(kwargs['buffer_start_iter']), int, IS_POSITIVE)
        self.sample_size = validate(int(kwargs['sample_size']), int, IS_POSITIVE)
        self.loss_fun = validate(kwargs['loss_fun'], str)
        self.eps_testing = validate(parse_bool(kwargs['eps_testing']), bool)
        self.Q_eps_dec_each_iter, self.Q_eps_start, self.Q_eps_end = tuple(
            validate(float(x), float, IS_BETWEEN_0_AND_1) for x in self.Q_sampling.split('_')[1:])
        self.loss = MSELoss() if self.global_loss_func == 'MSE' else BCEWithLogitsLoss()
        self.debug_first_train_iters = 50
        self.debug_train_iter_counter = 0
        self.seed = random.Random(123)
        self.train_counter = -1
        self.sample_strat, self.biased = ('sg', 'full') if opt.smarter_bin_sampling else (
            ('q_max', None) if opt.smart_bin_sampling else (None, 'biased'))
        self.buffer = BinBuffer(self.buffer_size, sample_strat=self.sample_strat, biased=self.biased,
                                no_trivial_pairs=opt.no_trivial_pairs)
        self.reward_calculator = RewardCalculator(opt.reward_calculator_mode, self.feat_map, self.calc_bound)
        self.dqn, self.dqn_tgt = [Q_network_v1(kwargs['encoder_type'], kwargs['embedder_type'],
                                               int(kwargs['in_dim']), int(kwargs['n_dim']), int(kwargs['n_layers']),
                                               kwargs['GNN_mode'], eval(kwargs['learn_embs']), eval(kwargs['layer_AGG_w_MLP']),
                                               int(kwargs['Q_mode']), kwargs['Q_act'], self.reward_calculator,
                                               self._environment)]*2
        self.forward_config_dict = self.get_forward_config_dict(self.restore_bidomains, self.total_runtime,
                                                                self.recursion_threshold, self.q_signal)
        self.method_config_dict = self.get_method_config_dict(self.DQN_mode, self.regret_iters)
        self.time_analysis = opt.time_analysis
        self.timer = OurTimer() if self.time_analysis else None
        self.val_every_iter, self.supervised_before, self.imitation_before = opt.val_every_iter, opt.supervised_before, opt.imitation_before

    #########################################################
    # Forward Procedure
    #########################################################
    def forward(self, ins, batch_data, iter=None, cur_id=None):
        if self.timer:
            self.timer.time_and_clear(f'forward iter {iter} start')

        forward_mode = self.get_forward_mode(iter)
        self.apply_forward_config(self.forward_config_dict[forward_mode])

        methods = opt.val_method_list if forward_mode == TEST_MODE else (
            ['dqn'] if forward_mode == TRAIN_MODE else ['mcspv2'])
        for method in methods:
            # run forward model
            self.apply_method_config(self.method_config_dict[method])
            pair_list, state_init_list = self._preprocess_forward(ins, batch_data, cur_id)
            self._forward_batch(pair_list, state_init_list)

        if forward_mode != TEST_MODE:
            # run loss function
            self.apply_method_config(self.method_config_dict['dqn'])
            loss = self._loss_wrapper(forward_mode)

            if self._tgt_net_sync_itr():
                self._sync_tgt_networks()
        else:
            loss = None

        return loss

    def _forward_batch(self, pair_list, state_init_list):
        for pair, state_init in zip(pair_list, state_init_list):
            # run the search procedure
            search_tree = self._forward_single_tree(state_init)

            # extend buffer with search tree
            if self.training:
                buffer_entry_list = self.search_tree2buffer_entry_list(search_tree, pair)
                self.buffer.extend(buffer_entry_list)

    def _forward_single_tree(self, state_init):
        # initializations
        search_stack = StackHeap()
        search_stack.add(state_init, 0)
        search_tree = SearchTree(root=state_init)
        since_last_update_count = 0

        timer = OurTimer()
        recursion_count = 0

        incumbent_list = [['incumbent', 'recursion_count', 'time']]

        # run the search code
        incumbent = {}
        incumbent_local_len = len(incumbent)
        if opt.load_model is not None:
            saver.log_info('search starting!')
        while len(search_stack) != 0:
            recursion_count += 1
            if recursion_count%500 == 0 and opt.load_model is not None:
                saver.log_info(f'on iteration {recursion_count}:\t{len(incumbent)}')

            # pop from stack
            cur_state, promise, incumbent_local_len, since_last_update_count = \
                self.sample_search_stack(
                    search_stack,
                    incumbent_local_len,
                    since_last_update_count)

            # update incumbent
            if self.is_better_solution(cur_state, len(incumbent)):
                # print(f'on iteration {recursion_count}:\t{len(incumbent)}')
                incumbent = deepcopy(cur_state.nn_map)
                incumbent_list.append(
                    [incumbent, recursion_count, timer.get_duration()])

            if opt.time_analysis:
                self.timer.time_and_clear(f'recursion {recursion_count} update incumbent')

            # check for exit conditions
            if self.exit_condition(recursion_count, timer, since_last_update_count):
                break

            action_space_data = \
                self.get_action_space_data_wrapper(cur_state, is_mcsp=self.get_is_mcsp())

            pruned, search_stack = self.prune_condition(cur_state, action_space_data, incumbent,
                                                        search_stack, search_tree)
            if pruned:
                if self.training and self.search_path:
                    break
                else:
                    continue

            action_edge, next_state, promise_tuple, q_pred = self._forward_single_edge(cur_state, action_space_data,
                                                                                       recursion_count)

            if opt.time_analysis:
                self.timer.time_and_clear(f'recursion {recursion_count} _forward_single_edge')

            # update search stack
            promise, promise_new_state = promise_tuple

            (v, w) = action_edge.action
            # assert v not in cur_state.pruned_actions.l2r or cur_state.pruned_actions.l2r[v] != w
            # saver.log_info(action_edge.action)
            cur_state.prune_action(v=v, w=w)

            search_stack.add(cur_state, promise)
            search_stack.add(next_state, promise_new_state)
            discount = self.reward_calculator.discount
            # update RL search tree
            search_tree.link_states(
                cur_state, action_edge, next_state, q_pred, discount)

            # raise Exception('finished one forward pass')
        incumbent_list.append(
            [incumbent, recursion_count, timer.get_duration()])

        self.post_process(search_tree, incumbent_list)

        return search_tree, incumbent

    def _forward_single_edge(self, state, action_space_data, recursion_count):

        # estimate the q values
        if 'fixedv' in self.DQN_mode:
            mode = self.DQN_mode.split('_')[-1]
            action = self.fixed_v(action_space_data, state, mode)
            if self.DQN_mode == 'fixedv_mcsprl':
                self.update_lgrade_rgrade(state, action)

            promise = 0
            q_vec, q_vec_idx, is_q_vec_idx_argmax = \
                self.get_mcsp_promise2q_vec(promise)
        else:
            q_vec = self.compute_q_vec(state, action_space_data).detach()

            assert q_vec.size(0) == len(action_space_data.action_space[0]) == len(
                action_space_data.action_space[1])
            # greedily select the action using q values
            action, q_vec_idx, is_q_vec_idx_argmax = \
                self._rank_actions_from_q_vec(q_vec, state,
                                              action_space_data)  # TODO: no graph_data

        # compute the next_state and any accompanying metadata
        action_edge, next_state = self._environment(state, action,
                                                    recursion_count=recursion_count,
                                                    real_update=True)

        # compute promise (MUST BE AFTER ENVIRONMENT!)
        promise_tuple = self.find_promise(next_state, action_space_data)

        # return selected action and next state
        return action_edge, next_state, promise_tuple, q_vec[q_vec_idx]

    def get_mcsp_promise2q_vec(self, promise):
        q_vec = torch.tensor([promise, 0])
        q_vec_idx = 0
        is_q_vec_idx_argmax = False
        assert opt.promise_mode in ['diverse']  # 'P', 'root', 'random', 'diverse'] # D not supported
        return q_vec, q_vec_idx, is_q_vec_idx_argmax

    def update_lgrade_rgrade(self, state, action):
        v, w = action
        # connected
        for bd in unroll_bidomains(state.natts2bds):
            self.lgrade[v] += min(len(bd.left), len(bd.right))
            self.rgrade[w] += min(len(bd.left), len(bd.right))
        # unconnected bidomains: we do this because the unconnected bd is usually too big to fit in memory
        natts2g2abd_sg_nids = \
            get_natts2g2abd_sg_nids(state.natts2g2nids, state.natts2bds, state.nn_map)
        for natts, g2nids in state.natts2g2nids.items():
            if natts in natts2g2abd_sg_nids:
                g2abd_sg_nids = natts2g2abd_sg_nids[natts]
                min_lr = \
                    min(
                        len(g2nids['g1']) - len(g2abd_sg_nids['g1']),
                        len(g2nids['g2']) - len(g2abd_sg_nids['g2'])
                    )
            else:
                min_lr = min(len(g2nids['g1']), len(g2nids['g2']))
            self.lgrade[v] += min_lr
            self.rgrade[w] += min_lr

    #########################################################
    # Utility Functions
    #########################################################
    def get_forward_mode(self, iter):
        if iter%self.val_every_iter == 0:
            forward_mode = TEST_MODE
        else:
            self.train_counter += 1
            if self.train_counter < self.supervised_before:
                forward_mode = PRETRAIN_MODE
            elif self.train_counter < self.imitation_before:
                forward_mode = IMITATION_MODE
            else:
                forward_mode = TRAIN_MODE

        return forward_mode

    def get_forward_config_dict(self, restore_bidomains, total_runtime, recursion_threshold, q_signal):
        self.q_signal = None
        self.recursion_threshold = None
        self.restore_bidomains = None
        self.total_runtime = None
        self.search_path = None
        self.no_pruning = None

        total_runtime_test = validate(opt.total_runtime, int, IS_POSITIVE, None)
        recursion_threshold_test = validate(opt.recursion_threshold, int, IS_POSITIVE, None)

        forward_config_dict = {
            PRETRAIN_MODE:
                ForwardConfig(
                    10,
                    None,
                    'LB',
                    True,
                    True,
                    False,
                    True,
                ),
            IMITATION_MODE:
                ForwardConfig(
                    total_runtime,
                    recursion_threshold,
                    q_signal,
                    restore_bidomains,
                    False,
                    True,
                    True
                ),
            TRAIN_MODE:
                ForwardConfig(
                    total_runtime,
                    recursion_threshold,
                    q_signal,
                    restore_bidomains,
                    False,
                    True,
                    True
                ),
            TEST_MODE:
                ForwardConfig(
                    total_runtime_test,
                    recursion_threshold_test,
                    q_signal,
                    False,
                    False,
                    False,
                    False
                )
        }

        return forward_config_dict

    def get_method_config_dict(self, DQN_mode, regret_iters):
        self.DQN_mode = None
        self.regret_iters = None

        method_config_dict = {
            'dqn':
                MethodConfig(
                    DQN_mode,
                    regret_iters
                ),
            'mcspv2':
                MethodConfig(
                    'fixedv_mcsp',
                    None
                ),
            'mcsprl':
                MethodConfig(
                    'fixedv_mcsprl',
                    None
                )
        }
        return method_config_dict

    def apply_forward_config(self, forward_config):
        self.total_runtime = forward_config.total_runtime
        self.recursion_threshold = forward_config.recursion_threshold
        self.q_signal = forward_config.q_signal
        self.restore_bidomains = forward_config.restore_bidomains
        self.no_pruning = forward_config.no_pruning
        self.search_path = forward_config.search_path
        self.training = forward_config.training

    def apply_method_config(self, method_config):
        self.DQN_mode = method_config.DQN_mode
        self.regret_iters = method_config.regret_iters

    def get_action_space_data_wrapper(self, state, is_mcsp=False):
        # get action space
        natts2bds_unexhausted = state.get_natts2bds_unexhausted(with_bids=True)

        action_space_size_unexhausted_unpruned = \
            self.get_action_space_size_unexhausted_unpruned(natts2bds_unexhausted)
        bidomains = unroll_bidomains(natts2bds_unexhausted)

        if len(bidomains) == 0:
            action_space = self._get_empty_action_space()
            assert natts2bds_unexhausted == dict()
            natts2bds_pruned = natts2bds_unexhausted
        else:
            num_bds, num_nodes_degree, num_nodes_dqn = \
                self._get_prune_parameters(is_mcsp)
            increase_degree = int(min(1, num_nodes_degree*1.4142135623))
            increase_dqn = int(min(1, num_nodes_dqn*1.4142135623))
            action_space = [[], [], []]

            # print(f'Here I am {len(bidomains)} ismcsp {is_mcsp}: l ({[len(bd.left) for bd in bidomains]}) r ({[len(bd.right) for bd in bidomains]})')
            # print(f'State nn_map: {state.nn_map}')
            # print(f'State pruned states: {state.pruned_actions}')

            while (len(action_space[0]) == 0):  # TODO: make this more efficient
                num_nodes_degree += increase_degree
                num_nodes_dqn += increase_dqn

                # prune topK adjacent bidomains
                bds_pruned = \
                    self._prune_topk_bidomains(bidomains, num_bds)

                # prune top(L1/#bidomains) nodes
                bds_pruned = \
                    self._prune_topk_nodes(bds_pruned, num_nodes_degree, state, 'deg')

                natts2bds_pruned = defaultdict(list)
                for bd in bds_pruned:
                    natts2bds_pruned[bd.natts].append(bd)

                # get possible node pairs from list of bidomains
                # all combinations of nodes from bd.left and bd.right for all bds
                if is_mcsp and len(state.nn_map) == 0:
                    # bds_pruned_i = invert_bds(natts2bds_pruned, state)
                    action_space = self._get_empty_action_space()
                    break
                else:
                    action_space = self._format_action_space(bds_pruned, state)

        # put action space into a wrapper
        action_space_data = \
            ActionSpaceDataScalable(
                action_space,
                natts2bds_pruned,
                action_space_size_unexhausted_unpruned
            )

        return action_space_data

    def _get_prune_parameters(self, is_mcsp):
        if is_mcsp:
            num_bds = 1
            num_nodes_degree = float('inf')
            num_nodes_dqn = float('inf')
        else:
            num_bds = float('inf') if opt.num_bds_max < 0 else opt.num_bds_max
            num_nodes_degree = float('inf') if opt.num_nodes_degree_max < 0 else opt.num_nodes_degree_max
            num_nodes_dqn = float('inf') if opt.num_nodes_dqn_max < 0 else opt.num_nodes_dqn_max

        return num_bds, num_nodes_degree, num_nodes_dqn

    def _prune_topk_bidomains(self, bidomains, num_bds):
        # select for topk bidomains
        prune_flag = len(bidomains) > num_bds
        if prune_flag:
            bds_pruned = \
                self._filter_topk_bds_by_size(bidomains, num_bds)
        else:
            bds_pruned = bidomains

        return bds_pruned

    def _prune_topk_nodes(self, bidomains, num_nodes, state, method):
        # get L value (max number of nodes in each bidomain)
        num_nodes_per_bd = num_nodes//len(bidomains)

        # prune for topl nodes
        bds_pruned, bdids_pruned = [], []
        for k, bd in enumerate(bidomains):
            prune_flag_l = len(bd.left) > num_nodes_per_bd
            prune_flag_r = len(bd.right) > num_nodes_per_bd
            if prune_flag_l:
                if method == 'deg':
                    left_domain = self._filter_topk_nodes_by_degree(
                        bd.left, num_nodes_per_bd, state.g1)
                else:
                    assert False
            else:
                left_domain = bd.left
            if prune_flag_r:
                if method == 'deg':
                    right_domain = self._filter_topk_nodes_by_degree(
                        bd.right, num_nodes_per_bd, state.g2)
                else:
                    assert False
            else:
                right_domain = bd.right

            if prune_flag_l or prune_flag_r:
                bds_pruned.append(Bidomain(left_domain, right_domain, None, bd.natts, bd.bid))
            else:
                bds_pruned.append(bd)
        return bds_pruned

    def _get_empty_action_space_data(self, state):
        natts2bds_unexhausted = state.get_natts2bds_unexhausted(with_bids=True)
        action_space_data = \
            ActionSpaceDataScalable(self._get_empty_action_space(), natts2bds_unexhausted, None)
        return action_space_data

    def _get_empty_action_space(self):
        return [[], [], []]

    def _format_action_space(self, bds, state):
        left_indices = []
        right_indices = []
        bd_indices = []
        # soft matching: possibly give diff scores to pairs
        for bd in bds:
            for v in bd.left:
                for w in bd.right:
                    # bds only contain unexhausted nodes NOT unexhausted edges
                    #   -> MUST check here to ensure nodes aren't revisited!
                    if v in state.pruned_actions.l2r and w in state.pruned_actions.l2r[v]:
                        continue
                    left_indices.append(v)
                    right_indices.append(w)
                    bd_indices.append(bd.bid)

        action_space = [left_indices, right_indices, bd_indices]
        assert len(left_indices) == len(right_indices) == len(bd_indices)
        return action_space

    def _filter_topk_bds_by_size(self, bidomains, num_bds_max):
        degree_list = np.array([max(len(bd.left), len(bd.right)) for bd in bidomains])
        if opt.inverse_bd_size_order:
            degree_list_sorted = degree_list.argsort(kind='mergesort')[::-1]
        else:
            degree_list_sorted = degree_list.argsort(kind='mergesort')
        indices = degree_list_sorted[:num_bds_max]
        return [bidomains[idx] for idx in indices]

    def _filter_topk_nodes_by_degree(self, all_nodes, num_nodes_max, g):
        nodes = list(all_nodes)
        degree_list = np.array([g.degree[node] for node in nodes])
        indices = degree_list.argsort(kind='mergesort')[-num_nodes_max:][::-1]
        return [nodes[idx] for idx in indices]

    ##########################################################
    # Utility functions (forward procedure)
    ##########################################################

    def _preprocess_forward(self, ins, batch_data, cur_id):
        offset = 0
        state_init_list = []
        pair_list = batch_data.pair_list
        for pair in pair_list:
            # set up general input data
            g1, g2 = pair.g1, pair.g2  # Get graph from a pair
            ins_g1, ins_g2, offset = self.compute_ins(g1, g2, ins,
                                                      offset)  # Compute their tensors from the contiguous one (ins) and update offset
            edge_index1, edge_index2 = create_edge_index(g1), create_edge_index(
                g2)  # Create indexes for the edges of each graph
            adj_list1, adj_list2 = create_adj_set(g1), create_adj_set(g2)  # Create adjacency matrix for each graph
            nn_map = {}
            nn_map_neighbors = {'g1': set(), 'g2': set()}

            ######################################  Adds node id of each nodes of each graph using its hash as key
            natts2g2nids = defaultdict(lambda: defaultdict(set))
            for nid in range(g1.number_of_nodes()):
                natts2g2nids[get_natts_hash(g1.nodes[nid])]['g1'].add(nid)
            for nid in range(g2.number_of_nodes()):
                natts2g2nids[get_natts_hash(g2.nodes[nid])]['g2'].add(nid)
            natts2bds = {}

            natts2g2abd_sg_nids = get_natts2g2abd_sg_nids(natts2g2nids, natts2bds, nn_map)  # TODO: understand this
            ######################################
            MCS_size_UB = self.calc_bound_helper(natts2bds, natts2g2nids,
                                                 natts2g2abd_sg_nids)  # Calculate MCS upper bound

            # set up special input data
            mcsp_vec = None
            if self.DQN_mode in ['fixedv_mcsp', 'fixedv_mcsprl'] or opt.use_mcsp_policy:
                mcsp_vec = self.get_mcsp_vec(g1, g2)  # Use MCS heuristic based on node's degree
            assert self.DQN_mode not in ['pca', 'sgw', 'mcsp_degree']  # not supported

            # create input data object
            torch.set_printoptions(profile="full")
            state_init = StateNode(ins_g1,
                                   ins_g2,
                                   nn_map,
                                   nn_map_neighbors,
                                   natts2bds,
                                   natts2g2nids,
                                   edge_index1,
                                   edge_index2,
                                   adj_list1,
                                   adj_list2,
                                   g1, g2,
                                   cur_id,
                                   mcsp_vec,
                                   MCS_size_UB)
            state_init_list.append(state_init)
        assert len(pair_list) == len(state_init_list)
        return pair_list, state_init_list

    def compute_ins(self, g1, g2, ins, offset):
        M, N = g1.number_of_nodes(), g2.number_of_nodes()
        ins_g1, ins_g2 = ins[offset:offset + M], ins[offset + M:offset + N + M]
        offset += (N + M)  # used for grabbing the right input embeddings
        return ins_g1, ins_g2, offset

    def _tgt_net_sync_itr(self):
        valid_iteration = self.train_counter%self.sync_target_frames == 0
        return valid_iteration

    def _sync_tgt_networks(self):
        self.dqn_tgt.load_state_dict(self.dqn.state_dict())

    def get_mcsp_vec(self, g1, g2):
        deg_vec_g1 = np.array(list(g1.degree[j] for j in range(g1.number_of_nodes())))
        deg_vec_g2 = np.array(list(g2.degree[j] for j in range(g2.number_of_nodes())))
        self.lgrade = deg_vec_g1/(np.max(deg_vec_g1) + 2)
        self.rgrade = deg_vec_g2/(np.max(deg_vec_g2) + 2)
        mcsp_vec = McspVec(deg_vec_g1, deg_vec_g2)
        return mcsp_vec

    ##########################################################
    # Utility functions (forward single tree procedure)
    ##########################################################
    def sample_search_stack(self, search_stack, incumbent_local_len, since_last_update_count):
        if self.regret_iters is None:
            method = 'stack'
        else:
            if since_last_update_count > self.regret_iters:
                method = 'heap'
            else:
                method = 'stack'

        cur_state, promise = search_stack.pop_task(method)

        if self.is_better_solution(cur_state, incumbent_local_len):
            incumbent_local_len = len(cur_state.nn_map)
            since_last_update_count = 0
        else:
            if method == 'heap':
                since_last_update_count = 0  # -(len(incumbent) - len(cur_state.nn_map))
                incumbent_local_len = len(cur_state.nn_map)
            elif method == 'stack':
                since_last_update_count += 1
            else:
                assert False
        return cur_state, promise, incumbent_local_len, since_last_update_count

    def is_better_solution(self, cur_state, incumbent_len):
        return len(cur_state.nn_map) > incumbent_len

    def exit_condition(self, recursion_count, timer, since_last_update_count):
        # exit search if recursion threshold
        recursion_thresh = (
                                   self.recursion_threshold is not None) and recursion_count > self.recursion_threshold
        timout_thresh = self.total_runtime is not None and timer.get_duration() > self.total_runtime
        return recursion_thresh and timout_thresh

    def prune_condition(self, cur_state, action_space_data, incumbent,
                        search_stack, search_tree):
        # compute bound
        bound = self.calc_bound(cur_state)

        # check prune conditions
        empty_action_space = len(action_space_data.natts2bds_unexhausted) == 0
        bnb_condition = len(cur_state.nn_map) + bound <= len(incumbent)
        return empty_action_space or ((not self.no_pruning) and bnb_condition), search_stack

    def post_process(self, search_tree, incumbent_list):
        search_tree.assign_v_search_tree(self.reward_calculator.discount)
        if not self.training and opt.load_model is not None:
            incumbent_end, recursion_iter_end, time_end = incumbent_list[-1]
            saver.log_info('=========================')
            saver.log_info(f'length of largest incumbent: {len(incumbent_end)}')
            saver.log_info(f'iteration at end: {recursion_iter_end}')
            saver.log_info(f'time at end: {time_end}')
            saver.log_info('=========================')

    def search_tree2buffer_entry_list(self, search_tree, pair):
        if self.training:
            g1, g2 = pair.g1, pair.g2
            if opt.exclude_root:
                buffer_entry_list = [BufferEntry(edge, g1, g2, search_tree) for edge in search_tree.edges if
                                     edge.state_prev.action_prev is not None]
            else:
                buffer_entry_list = [BufferEntry(edge, g1, g2, search_tree) for edge in search_tree.edges]
        else:
            buffer_entry_list = None

        return buffer_entry_list

    def calc_bound(self, state, exhaust_revisited_nodes=True):
        natts2g2nids = state.natts2g2nids
        natts2bds = state.natts2bds
        nn_map = state.nn_map

        # MUST USE UNFILTERED natts2bds OTHERWISE WILL DROP THE EXHAUSTED NODES!
        natts2g2abd_sg_nids = \
            get_natts2g2abd_sg_nids(natts2g2nids, natts2bds, nn_map)

        if exhaust_revisited_nodes:
            assert state is not None
            natts2bds_filtered = state.get_natts2bds_abd_unexhausted()
        else:
            natts2bds_filtered = state.natts2bds

        return \
            self.calc_bound_helper(natts2bds_filtered, natts2g2nids, natts2g2abd_sg_nids)

    def calc_bound_helper(self, natts2bds, natts2g2nids, natts2g2abd_sg_nids):

        bd_lens = []
        # adjacent bidomains
        for bd in unroll_bidomains(natts2bds):
            bd_lens.append((len(bd.left), len(bd.right)))

        # disconnected bidomains
        for natts, g2nids in natts2g2nids.items():
            if natts in natts2g2abd_sg_nids:
                #: NOTE: NO GUARANTEE THAT 0 len unconnected bds are not appended here!
                g2abd_sg_nids = natts2g2abd_sg_nids[natts]
                bd_lens.append(
                    (
                        len(g2nids['g1']) - len(g2abd_sg_nids['g1']),
                        len(g2nids['g2']) - len(g2abd_sg_nids['g2'])
                    )
                )
            else:
                bd_lens.append((len(g2nids['g1']), len(g2nids['g2'])))

        bound = 0
        for left_len, right_len in bd_lens:
            bound += min(left_len, right_len)
        return bound

    ##########################################################
    # Utility functions (forward single edge procedure)
    ##########################################################
    def compute_q_vec(self, state, action_space_data):
        # estimate the q values
        if len(action_space_data.action_space[0]) > 1 or opt.plot_final_tree:
            # we want all q_pred if we are plotting tree!
            q_vec = self._Q_network(state, action_space_data, tgt_network=False, detach_in_chunking_stage=True)
        else:
            q_vec = torch.ones(1)
        return q_vec

    def _Q_network(self, state, action_space_data, tgt_network=False, detach_in_chunking_stage=False):
        dqn_input = DQNInput(state, action_space_data, self.restore_bidomains)
        if tgt_network:
            q_vec = self.dqn_tgt(dqn_input, detach_in_chunking_stage)
        else:
            q_vec = self.dqn(dqn_input, detach_in_chunking_stage)
        return q_vec

    def fixed_v(self, action_space_data, state, mode):
        mcsp_vec = state.mcsp_vec
        last_v = state.last_v

        if mode == 'mcsp':
            lvec = mcsp_vec.ldeg
            rvec = mcsp_vec.rdeg
        elif mode == 'mcsprl':
            lvec = self.lgrade
            rvec = self.rgrade
        else:
            assert False

        # FAST b/c NODE RELATED
        # STATE -> ACTION instead of STATE -> Q_VEC -> ACTION
        # ASSUMES THAT EXHAUSTED NODES ARE ALREADY PRUNED IN BIDOMAIN!
        best_action = None
        best_v_score = -float('inf')
        for bd in unroll_bidomains(action_space_data.natts2bds_unexhausted):
            l_actions, r_actions = list(bd.left), list(bd.right)
            if last_v in l_actions:  # technically, this should be redundant?
                pruned_w = set() if last_v not in state.pruned_actions.l2r else state.pruned_actions.l2r[last_v]
                r_actions = list(set(r_actions) - pruned_w)
                w = r_actions[np.argmax(rvec[r_actions])]
                best_action = (last_v, w)
                break
            else:
                argmax_idx = np.argmax(lvec[l_actions])
                v_score = (lvec[l_actions])[argmax_idx]
                v = l_actions[argmax_idx]
                if v_score > best_v_score:
                    w = r_actions[np.argmax(rvec[r_actions])]
                    best_action = (v, w)

        assert best_action is not None
        return best_action

    def _compute_eps(self):
        eps = max(self.Q_eps_end, self.Q_eps_start -
                  self.train_counter*self.Q_eps_dec_each_iter)
        return eps

    def _rank_actions_from_q_vec(self, q_vec, state, action_space_data):
        if opt.use_mcsp_policy:
            assert False

        # compute epsilon (eps-greedy)
        if self.training:
            eps = self._compute_eps()
        else:
            if self.eps_testing:
                # same as AlphaGO, use Q_eps_end (to reduce overfitting)
                eps = self.Q_eps_end
                # print('\t val eps_testing', eps)
            else:
                eps = -1
                # print('\t val eps', eps)

        # epsilon greedy policy
        q_vec_idx_argmax = torch.argmax(q_vec, dim=0)
        if random.random() < eps or opt.randQ:  # and state.action_prev is None):
            # randomly pick an index
            q_vec_idx = int(random.random()*q_vec.size(0))

        else:
            q_vec_idx = q_vec_idx_argmax
        is_q_vec_idx_argmax = q_vec_idx == q_vec_idx_argmax

        action = (action_space_data.action_space[0][q_vec_idx],
                  action_space_data.action_space[1][q_vec_idx])
        return action, q_vec_idx, is_q_vec_idx_argmax

    def _environment(self, state, action, recursion_count=None, real_update=False):

        if opt.time_analysis:
            timer_env = OurTimer()
            timer_env.time_and_clear(f'environment starts')

        nn_map = deepcopy(state.nn_map)
        exhausted_v, exhausted_w = state.exhausted_v, state.exhausted_w
        if 'mcsp' in self.DQN_mode:
            pruned_actions = state.pruned_actions
        else:
            pruned_actions = None

        if opt.time_analysis:
            timer_env.time_and_clear(f'environment deepcopy')

        if opt.time_analysis:
            timer_env.time_and_clear(f'environment get_pruned_bidomains')

        v, w = action

        # apply action
        nn_map[v] = w

        # get next state
        g1, g2 = state.g1, state.g2
        natts2bds, nn_map_neighbors = \
            self._update_bidomains(
                g1, g2,
                action, nn_map, state.nn_map_neighbors,
                state.natts2bds, state.natts2g2nids
            )
        natts2g2abd_sg_nids = \
            get_natts2g2abd_sg_nids(state.natts2g2nids, natts2bds, nn_map)

        if opt.time_analysis:
            timer_env.time_and_clear(f'environment _update_bidomains')

        MCS_size_UB = \
            len(nn_map) + \
            self.calc_bound_helper(natts2bds, state.natts2g2nids, natts2g2abd_sg_nids)

        if self.restore_bidomains:
            exhausted_v = deepcopy(exhausted_v)
            exhausted_w = deepcopy(exhausted_v)
        else:
            exhausted_v = None
            exhausted_w = None

        # make new state node
        state.last_v = v
        next_state = StateNode(state.ins_g1,
                               state.ins_g2,
                               nn_map,
                               nn_map_neighbors,
                               natts2bds,
                               state.natts2g2nids,
                               state.edge_index1,
                               state.edge_index2,
                               state.adj_list1,
                               state.adj_list2,
                               state.g1, state.g2,
                               state.cur_id,
                               state.mcsp_vec,
                               MCS_size_UB,
                               tree_depth=state.tree_depth + 1,
                               num_steps=state.num_steps + 1,
                               exhausted_v=exhausted_v,
                               exhausted_w=exhausted_w)
        if opt.time_analysis:
            timer_env.time_and_clear(f'environment _init StateNode')

        reward = self.reward_calculator.compute_reward(v, w, g1, g2, state, next_state)

        action_edge = ActionEdge(
            action, None, reward,
            deepcopy(pruned_actions),
            deepcopy(exhausted_v),
            deepcopy(exhausted_w))

        if opt.time_analysis:
            timer_env.time_and_clear(f'environment end')
            timer_env.print_durations_log()

        return action_edge, next_state

    def _update_bidomains(self, g1, g2, action, nn_map, nn_map_neighbors, natts2bds, natts2g2nids):

        '''
        LOGIC FOR BIDOMAIN COMPUTATION:

        STATIC DISCONNECTED BIDOMAIN, ubd0, MUST STILL EXIST!
        why? node labels for root node action space creation.
        how? create disconnected bidomains at start

        DYNAMIC DISCONNECTED BIDOMAIN MUST BE INFERRED!
        why? time complexity
        how? DVN will use dynamic adjacent bidomains to find dynamic disconnected bidomain

        DYNAMIC ADJACENT BIDOMAINS:
        2 cases:
            Let N = neighbors-nn_map
            1) adjacent bidomain (add node)
                Ni = (N - (N1 + N2 + ... +Ni-1)).intersect(abdi)
                    Ni == N.intersect(abd)
                abd1* = Ni - action
                abd0* = abdi - abd1* - action

            2) unconnected bidomain (add node)
                abd1 = N - (N1, N2, ..., Nk)
                    abd1 == ubd ⋂ N
        BUT nodes have labels!!:
            1*)
                Ni = (N - (N1 + N2 + ... +Ni-1)).intersect(abdi,j)
                    Ni == N.intersect(abd)
                abd1*,j = Ni - action                       <bitstring = <bitstringi>, 1> s.t. j is label
                abd0*,j = abdi,j - abd1*,j - action         <bitstring = <bitstringi>, 0>
            2*)
                let N(i) = neighbors with label i
                abd1,i
                = N(i) - (N1, N2, ..., Nk) - action
                = (N(i) - (N1, N2, ..., Nk))(i) - action    <bitstring = 0, ..., 0, 1_i> s.t. i is label

        DYNAMIC DISCONNECTED BIDOMAINS:
        ubd(i) = ubd0(i) - nn_map - sum_j(abdsj)            <bitstring = 0, ..., 0> w. label i
        proof/
            ubd(i) = ubd0(i) - nn_map - N
            abds is DISJOINT UNION over N
            t.f. N = sum_j(abdsj) [X]
        '''

        # timer = None
        # if opt.time_analysis:
        #     timer = OurTimer()
        #     timer.time_and_clear(f'_update_bidomains starts')

        natts2bds_new = defaultdict(list)

        '''
            1*)
                Ni = (N - (N1 + N2 + ... +Ni-1)).intersect(abdi,j)
                    Ni == N.intersect(abd)
                abd1*,j = Ni - action                        <bitstring = <bitstringi>, 1> s.t. j is label
                abd0*,j = abdi,j - abd1*,j - action           <bitstring = <bitstringi>, 0>
                    abd2,j == abd - abd1,j - action
        '''
        u, v = action
        N_g1 = set(g1.neighbors(u)) - set(nn_map.keys())
        N_g2 = set(g2.neighbors(v)) - set(nn_map.values())

        nn_map_neighbors_new = \
            {
                'g1': nn_map_neighbors['g1'].union(N_g1) - {u},
                'g2': nn_map_neighbors['g2'].union(N_g2) - {v}
            }

        for natts, bidomains in natts2bds.items():
            for bidomain in bidomains:
                assert natts == bidomain.natts
                left, right = bidomain.left, bidomain.right
                Ni_g1, Ni_g2 = N_g1.intersection(left), N_g2.intersection(right)
                left_1, right_1 = Ni_g1 - {u}, Ni_g2 - {v}
                left_0, right_0 = left - Ni_g1 - {u}, right - Ni_g2 - {v}
                if len(left_1) > 0 and len(right_1) > 0:
                    natts2bds_new[natts].append(
                        Bidomain(left_1, right_1, None, natts))
                if len(left_0) > 0 and len(right_0) > 0:
                    natts2bds_new[natts].append(
                        Bidomain(left_0, right_0, None, natts))

                # remaining nodes will not belong to any adjacent bidomain!
                # => partition from unadjacent bidomain
                N_g1, N_g2 = N_g1 - Ni_g1, N_g2 - Ni_g2

        '''
            2*)
                let N(i) = neighbors with label i
                abd1,i
                = unconnected[N(i) - (N1, N2, ..., Nk)] - action
                = unconnected[(N(i) - (N1, N2, ..., Nk))(i)] - action    <bitstring = 0, ..., 0, 1_i> s.t. i is label
        '''
        for natts, g2nids in natts2g2nids.items():
            nid_natts_all_g1, nid_natts_all_g2 = g2nids['g1'], g2nids['g2']
            left_1_natts = N_g1.intersection(nid_natts_all_g1) - {u} - nn_map_neighbors['g1']
            right_1_natts = N_g2.intersection(nid_natts_all_g2) - {v} - nn_map_neighbors['g2']
            if len(left_1_natts) > 0 and len(right_1_natts) > 0:
                natts2bds_new[natts].append(
                    Bidomain(left_1_natts, right_1_natts, None, natts))
        return natts2bds_new, nn_map_neighbors_new

    def find_promise(self, next_state, action_space_data):
        promise = action_space_data.action_space_size_unexhausted_unpruned
        promise_new_state = \
            self.get_action_space_size_unexhausted_unpruned(
                next_state.get_natts2bds_unexhausted(with_bids=True)
            )
        promise, promise_new_state = -promise, -promise_new_state
        promise_tuple = (promise, promise_new_state)
        return promise_tuple

    def get_is_mcsp(self):
        return self.DQN_mode in ['fixedv_mcsp', 'fixedv_mcsprl']

    ##########################################################
    # Loss Function
    ##########################################################
    def _loss_wrapper(self, forward_mode):
        if forward_mode == PRETRAIN_MODE:
            loss = self._loss(self.buffer.sample(self.sample_size))
            self.buffer.empty()
        elif forward_mode in [IMITATION_MODE, TRAIN_MODE]:
            if self.train_counter < self.buffer_start_iter:
                loss = None
            else:
                loss = self._loss(self.buffer.sample(self.sample_size))
        else:
            assert False

        return loss

    def _loss(self, buffer_entry_list):
        assert not self.get_is_mcsp()
        if len(buffer_entry_list) == 0:
            # edge case 1: we are still buffering
            loss = None
        else:
            loss = torch.tensor(0.0, device=opt.device)
            for buffer_entry in buffer_entry_list:
                self._process_buffer_entry(buffer_entry)
                loss += self._batched_loss([buffer_entry])
            loss /= len(buffer_entry_list)
        return loss

    def _batched_loss(self, buffer_entry_list):
        assert len(buffer_entry_list) != 0
        q_vec_pred, q_vec_true = [], []
        for buffer_entry in buffer_entry_list:
            loss_input_li = self.get_loss_input_li(buffer_entry)
            for loss_input in loss_input_li:
                q_pred, q_true = loss_input
                q_vec_pred.append(q_pred)
                q_vec_true.append(q_true)
        q_vec_pred = torch.stack(tuple(q_vec_pred)).view(-1)
        q_vec_true = torch.stack(tuple(q_vec_true)).view(-1)
        q_vec_pred, q_vec_true = self._apply_post_batch_loss_normalization(q_vec_pred, q_vec_true)
        loss = torch.sum(self.loss(q_vec_pred, q_vec_true))

        self._log_loss(
            q_vec_pred, q_vec_true, buffer_entry_list[0].edge.state_prev, loss)
        return loss

    def _apply_post_batch_loss_normalization(self, q_pred, q_true):
        return self._normalize_vec(q_pred), self._normalize_vec(q_true)

    def _normalize_vec(self, q_in):
        if 'sm' in self.loss_fun:
            q_norm = softmax(q_in.view(-1), dim=0).view(-1)
            assert False
        else:
            q_norm = q_in.view(-1)
        return q_norm

    def _process_buffer_entry(self, buffer_entry):
        edge = buffer_entry.edge
        state = edge.state_prev
        next_state = edge.state_next
        self._push_edge_search_info_to_state(state, edge, next_state)

    def _push_edge_search_info_to_state(self, state, edge, next_state):
        state.pruned_actions, state.exhausted_v, state.exhausted_w = edge.pruned_actions, edge.exhausted_v, edge.exhausted_w
        next_state.pruned_actions, next_state.exhausted_v, next_state.exhausted_w = DoubleDict(), set(), set()

    def _get_pred_and_true_q(self, buffer_entry):
        g1, g2 = buffer_entry.g1, buffer_entry.g2
        edge = buffer_entry.edge
        state = edge.state_prev
        next_state = edge.state_next

        # compute q pred
        next_action_space_data = \
            self.get_action_space_data_wrapper(next_state, is_mcsp=self.get_is_mcsp())

        v, w = edge.action
        action_space_data = self._get_empty_action_space_data(state)
        action_space_data.filter_action_space_data(v, w)
        q_pred = self._Q_network(state, action_space_data, tgt_network=False)
        q_pred = torch.squeeze(q_pred)

        # compute q true
        if opt.device == 'cpu':
            reward = \
                torch.tensor(
                    self.reward_calculator.compute_reward_batch(
                        action_space_data.action_space, g1, g2, state, next_state),
                    device=opt.device
                ).type(torch.FloatTensor)
        else:
            reward = \
                torch.tensor(
                    self.reward_calculator.compute_reward_batch(
                        action_space_data.action_space, g1, g2, state, next_state),
                    device=opt.device
                ).type(torch.cuda.FloatTensor)

        q_next = \
            self._get_q_next(next_state, next_action_space_data)

        q_true = (reward + self.reward_calculator.discount*q_next).detach()

        return q_pred, q_true, state

    def _compute_tgt_q_max(self, next_state, next_action_space_data):
        is_empty_action_space = len(next_action_space_data.action_space[0]) == 0
        if is_empty_action_space:
            q_max = self.create_FloatTensor(0.0)
        else:
            q_vec = self._Q_network(next_state, next_action_space_data, tgt_network=True, detach_in_chunking_stage=True)
            q_max = torch.max(q_vec.to(opt.device).detach())

        return q_max

    def get_cum_reward(self, start_state, end_state, num_steps):
        discount = self.reward_calculator.discount**num_steps
        cum_reward = start_state.v_search_tree - discount*end_state.v_search_tree
        return cum_reward

    def _get_q_next(self, next_state, next_action_space_data):
        assert 'clamp' not in self.q_signal
        if 'fitted-tgt' in self.q_signal:
            # from state si, get the list (si, si+1, si+2, ..., sN)
            cur_end_state = next_state
            end_state_li = [next_state]
            while len(cur_end_state.action_next_list) > 0:
                if self.search_path:
                    assert len(cur_end_state.action_next_list) == 1
                    idx = 0
                else:
                    idx = int(self.seed.random()*len(cur_end_state.action_next_list))
                next_next_state = cur_end_state.action_next_list[idx].state_next
                end_state_li.append(next_next_state)
                cur_end_state = next_next_state

            # store as a list of (cum_reward, end_state, num_steps)
            if 'random-path' in self.q_signal:
                num_steps = int(self.seed.random()*len(end_state_li))
                end_state = end_state_li[num_steps]
                cum_reward = self.get_cum_reward(next_state, end_state, num_steps)
                cum_reward_end_state_li = [(cum_reward, end_state, num_steps)]
            elif 'leaf' in self.q_signal:
                num_steps = len(end_state_li) - 1
                end_state = end_state_li[num_steps]
                cum_reward = self.get_cum_reward(next_state, end_state, num_steps)
                cum_reward_end_state_li = [(cum_reward, end_state, num_steps)]
            else:
                assert False

            q_max = 0.0
            for cum_reward, end_state, num_steps in cum_reward_end_state_li:
                cur_next_state_action_space = self.get_action_space_data_wrapper(end_state, is_mcsp=self.get_is_mcsp())
                q_max_tgt = self._compute_tgt_q_max(end_state, cur_next_state_action_space)
                discount_factor = self.reward_calculator.discount**num_steps
                q_max += cum_reward + discount_factor*q_max_tgt
            q_max /= len(cum_reward_end_state_li)
        elif 'vanilla-tgt' in self.q_signal:
            # compute q_next via DQN
            q_max = self._compute_tgt_q_max(next_state, next_action_space_data)
        elif self.q_signal == 'LB':
            q_max = self.create_FloatTensor(next_state.v_search_tree)
        else:
            print(self.q_signal)
            assert False

        q_next = q_max.detach()
        return q_next

    def create_FloatTensor(self, li, requires_grad=False):
        if opt.device == 'cpu':
            tsr = torch.tensor(li, requires_grad=requires_grad, device=opt.device).type(torch.FloatTensor)
        else:
            tsr = torch.tensor(li, requires_grad=requires_grad, device=opt.device).type(torch.cuda.FloatTensor)
        return tsr

    def _log_loss(self, q_pred, q_true, state, loss):
        # logging
        if self.debug_train_iter_counter < self.debug_first_train_iters:
            saver.log_info(
                '\npred: {}\n' \
                'true: {}'.format(
                    ','.join(["{0:0.2f}".format(i) for i in q_pred.tolist()]),
                    ','.join(["{0:0.2f}".format(i) for i in q_true.tolist()]))
            )
            saver.log_info(
                'g1 size {} ' \
                'g2 size {} ' \
                'nn_map size {} ' \
                'loss_iter {:.3f} ' \
                'buffer size {} ' \
                'pair_id {}'.format(
                    state.g1.number_of_nodes(), state.g2.number_of_nodes(),
                    len(state.nn_map), loss.item(), len(self.buffer),
                    (state.g1.graph['gid'], state.g2.graph['gid']))
            )

        self.debug_train_iter_counter += 1

    def get_action_space_size_unexhausted_unpruned(self, natts2bds):
        as_size = 0
        for natts, bds in natts2bds.items():
            for bd in bds:
                as_size += len(bd)
        return as_size

    def get_loss_input_li(self, buffer_entry):
        loss_input_li = []
        q_pred, q_true, state = \
            self._get_pred_and_true_q(buffer_entry)
        loss_input_li.append(
            (q_pred, q_true)
        )
        return loss_input_li

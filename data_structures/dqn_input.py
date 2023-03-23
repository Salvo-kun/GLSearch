from __future__ import annotations
import torch
from .action_space import *
from options import opt
from .state_node import StateNode


class DQNInput:
    def __init__(self, state: StateNode, action_space_data: ActionSpace, restore_bidomains:bool):#, TIMER=None, recursion_count=None):
        self.restore_bidomains = restore_bidomains
        self.pair_id = (state.g1.graph['gid'], state.g2.graph['gid'])
        self.exhausted_v = state.exhausted_v
        self.exhausted_w = state.exhausted_w
        self.valid_edge_index1 = _get_valid_edge_index(
            state.edge_index1, state.adj_list1, state.exhausted_v)#, TIMER, recursion_count, '1')
        self.valid_edge_index2 = _get_valid_edge_index(
            state.edge_index2, state.adj_list2, state.exhausted_w)#, TIMER, recursion_count, '2')
        self.action_space_data = action_space_data
        self.state = state
        
        if not opt.scalable:
            assert isinstance(action_space_data, ActionSpaceData)
            for v,w in zip(action_space_data.action_space[0], action_space_data.action_space[1]):
                action_space_data = _get_empty_action_space_data(state)
                action_space_data.filter_action_space_data(v, w)

def _get_valid_edge_index(edge_index:torch.Tensor, adj_list:Dict[int, Set[int]], exhausted_nodes:Set[int]):#, TIMER, recursion_count, s):
    if len(exhausted_nodes) > 0:
        invalid_indices = [adj_list[v] for v in exhausted_nodes]
        valid_idx = list(set(range(edge_index.size(1))) - set().union(*invalid_indices))
        edge_index_pruned = torch.t(torch.t(edge_index)[valid_idx])
    else:
        edge_index_pruned = torch.t(torch.t(edge_index))

    return edge_index_pruned


def _get_empty_action_space_data(state):
    bds_unexhausted, bds_adjacent, bdids_adjacent = state.get_action_space_bds()
    action_space_data = ActionSpaceData([[], [], []], bds_unexhausted, bds_adjacent, bdids_adjacent)

    return action_space_data


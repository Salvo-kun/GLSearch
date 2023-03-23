from __future__ import annotations
from typing import Union
from options import opt
from .bidomain import *
from collections import defaultdict


class ActionSpaceData:
    def __init__(self, action_space, unexhausted_bds, bds, bdids):
        self.action_space = action_space
        self.unexhausted_bds = unexhausted_bds
        self.bds = bds
        self.bdids = bdids

    def filter_action_space_data(self, v, w):
        for i, bd in enumerate(self.bds):
            if v in bd.left and w in bd.right:
                self.action_space = [[v], [w], [self.bdids[i]]]
                return
        assert False


class ActionSpaceDataScalable:
    """
    Attributes:
        action_space: list of 3 lists : [
            indices of nodes in left graph,
            indices of nodes in right graph,
            indices of the bidomain
            ].
            All lists have N elements, and together they represent N possible node pairings between the 2 graphs.
        natts2bds_unexhausted: list all unpruned bidomains.
        action_space_size_unexhausted_unpruned: number of nodes in the unpruned bidomains.
    """

    def __init__(self,
                 action_space: List[List[int]],
                 natts2bds_unexhausted: Dict[str, list],
                 action_space_size_unexhausted_unpruned: int):
        self.action_space = action_space
        self.natts2bds_unexhausted = natts2bds_unexhausted
        self.action_space_size_unexhausted_unpruned = action_space_size_unexhausted_unpruned

    def filter_action_space_data(self, v, w):
        for bds in self.natts2bds_unexhausted.values():
            for bd in bds:
                if v in bd.left and w in bd.right:
                    self.action_space = [[v], [w], [bd.bid]]


# define ActionSpace type
ActionSpace = Union[ActionSpaceData, ActionSpaceDataScalable]


def get_action_space_data_wrapper(state: any, is_mcsp=False) -> ActionSpaceDataScalable:
    # get action space
    natts2bds_unexhausted = state.get_natts2bds_unexhausted(with_bids=True)

    action_space_size_unexhausted_unpruned = get_action_space_size_unexhausted_unpruned(natts2bds_unexhausted)
    bidomains = unroll_bidomains(natts2bds_unexhausted)  # get flat list of all unexhausted bidomains

    if len(bidomains) == 0:
        action_space = get_empty_action_space()
        assert natts2bds_unexhausted == dict()
        natts2bds_pruned = natts2bds_unexhausted
    else:
        num_bds, num_nodes_degree, num_nodes_dqn = get_prune_parameters(is_mcsp)
        increase_degree = int(min(1, num_nodes_degree*1.4142135623))
        increase_dqn = int(min(1, num_nodes_dqn*1.4142135623))
        action_space = [[], [], []]
        natts2bds_pruned = natts2bds_unexhausted

        while len(action_space[0]) == 0:  # TODO: make this more efficient
            num_nodes_degree += increase_degree
            num_nodes_dqn += increase_dqn

            # prune topK adjacent bidomains
            bds_pruned = prune_topk_bidomains(bidomains, num_bds)

            # prune top(L1/#bidomains) nodes
            bds_pruned = prune_topk_nodes(bds_pruned, num_nodes_degree, state, 'deg')

            natts2bds_pruned = defaultdict(list)
            for bd in bds_pruned:
                natts2bds_pruned[bd.natts].append(bd)

            # get possible node pairs from list of bidomains
            # all combinations of nodes from bd.left and bd.right for all bds
            if is_mcsp and len(state.nn_map) == 0:
                # bds_pruned_i = invert_bds(natts2bds_pruned, state)
                action_space = get_empty_action_space()
                break
            else:
                action_space = format_action_space(bds_pruned, state)

    # put action space into a wrapper
    action_space_data = ActionSpaceDataScalable(
        action_space,
        natts2bds_pruned,
        action_space_size_unexhausted_unpruned
    )

    return action_space_data


def get_action_space_size_unexhausted_unpruned(natts2bds: Dict[str, List[Bidomain]]) -> int:
    """Get size of action space = sum of the sizes of all bidomains for all node attributes"""
    as_size = 0
    for natts, bds in natts2bds.items():
        for bd in bds:
            as_size += len(bd)
    return as_size


def get_prune_parameters(is_mcsp: bool) -> (int | float, float, float):
    if is_mcsp:
        num_bds = 1
        num_nodes_degree = float('inf')
        num_nodes_dqn = float('inf')
    else:
        num_bds = float('inf') if opt.num_bds_max < 0 else opt.num_bds_max
        num_nodes_degree = float('inf') if opt.num_nodes_degree_max < 0 else opt.num_nodes_degree_max
        num_nodes_dqn = float('inf') if opt.num_nodes_dqn_max < 0 else opt.num_nodes_dqn_max

    return num_bds, num_nodes_degree, num_nodes_dqn


def get_empty_action_space():
    return [[], [], []]


def get_empty_action_space_data(state):
    natts2bds_unexhausted = state.get_natts2bds_unexhausted(with_bids=True)
    action_space_data = \
        ActionSpaceDataScalable(get_empty_action_space(), natts2bds_unexhausted, None)
    return action_space_data


def format_action_space(bds: list, state) -> [List[int], List[int], List[int]]:
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


def prune_topk_nodes(bidomains: List[Bidomain], num_nodes: int, state: StateNode, method: str) -> List[Bidomain]:
    # get L value (max number of nodes in each bidomain)
    num_nodes_per_bd = num_nodes//len(bidomains)

    # prune for topl nodes
    bds_pruned, bdids_pruned = [], []
    for k, bd in enumerate(bidomains):
        prune_flag_l = len(bd.left) > num_nodes_per_bd
        prune_flag_r = len(bd.right) > num_nodes_per_bd
        if prune_flag_l:
            if method == 'deg':
                left_domain = filter_topk_nodes_by_degree(
                    bd.left, num_nodes_per_bd, state.g1)
            else:
                assert False
        else:
            left_domain = bd.left
        if prune_flag_r:
            if method == 'deg':
                right_domain = filter_topk_nodes_by_degree(
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


def prune_topk_bidomains(bidomains: List[Bidomain], num_bds: int) -> List[Bidomain]:
    """ If there are more than num_bds bidomains, keep only the num_bds larger bidomains"""
    prune_flag = len(bidomains) > num_bds
    if prune_flag:
        bds_pruned = filter_topk_bds_by_size(bidomains, num_bds)
    else:
        bds_pruned = bidomains

    return bds_pruned


def filter_topk_bds_by_size(bidomains: List[Bidomain], num_bds_max: int) -> List[Bidomain]:
    # TODO @Hilicot shouldn't we use the min() instead of the max()? (pick the bidomains with the largest possible isomorphic graph pairs)
    degree_list = np.array([max(len(bd.left), len(bd.right)) for bd in bidomains])
    if opt.inverse_bd_size_order:
        degree_list_sorted = degree_list.argsort(kind='mergesort')[::-1]
    else:
        degree_list_sorted = degree_list.argsort(kind='mergesort')
    indices = degree_list_sorted[:num_bds_max]
    return [bidomains[idx] for idx in indices]


def filter_topk_nodes_by_degree(all_nodes: Set[int], num_nodes_max: int, g: Graph) -> List[int]:
    nodes = list(all_nodes)
    degree_list = np.array([g.degree[node] for node in nodes])
    indices = degree_list.argsort(kind='mergesort')[-num_nodes_max:][::-1]
    return [nodes[idx] for idx in indices]

from __future__ import annotations
from torch import Tensor
from options import opt
from .common import DoubleDict
from collections import defaultdict
from .bidomain import *


class StateNode(object):
    def __init__(self,
                 ins_g1: Tensor, ins_g2: Tensor,
                 nn_map: dict,
                 bidomains_or_nn_map_neighbors: Dict[str, Set[int]],
                 abidomains_or_natts2bds: dict,
                 ubidomains_or_natts2g2nids: dict,
                 edge_index1: Tensor, edge_index2: Tensor,
                 adj_list1: dict, adj_list2: dict,
                 g1: Graph, g2: Graph,
                 cur_id: int,
                 mcsp_vec: Optional[McspVec],
                 MCS_size_UB: int,
                 explore_n_pairs=None, pruned_actions=None, exhausted_v=None, exhausted_w=None,
                 tree_depth=0, num_steps=0, cum_reward=0):
        self.ins_g1 = ins_g1
        self.ins_g2 = ins_g2
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.adj_list1 = adj_list1
        self.adj_list2 = adj_list2
        self.g1 = g1
        self.g2 = g2
        self.x1 = None
        self.x2 = None
        self.nn_map = nn_map
        if opt.scalable:
            self.nn_map_neighbors = bidomains_or_nn_map_neighbors
            self.natts2bds = abidomains_or_natts2bds
            self.natts2g2nids = ubidomains_or_natts2g2nids
        else:
            self.bidomains = bidomains_or_nn_map_neighbors
            self.abidomains = abidomains_or_natts2bds
            self.ubidomains = ubidomains_or_natts2g2nids
        self.q_vec = None
        self.recompute = False

        # for additional pruning
        self.pruned_actions = DoubleDict() if pruned_actions is None else pruned_actions
        self.exhausted_v = set() if exhausted_v is None else exhausted_v
        self.exhausted_w = set() if exhausted_w is None else exhausted_w

        # for search tree
        self.action_next_list = []
        self.action_prev = None

        self.MCS_size_UB = MCS_size_UB
        self.cum_reward = cum_reward
        self.v_search_tree = 0  # exhausted_q_max_LB

        self.explore_n_pairs = explore_n_pairs  # thresholds number of times we backtrack
        self.nid = None  # for search_tree.nxgraph
        self.tree_depth = tree_depth
        self.num_steps = num_steps
        self.cur_id = cur_id
        self.mcsp_vec = mcsp_vec
        self.last_v = None

    def compute_reward_from(self, start_node, discount):
        reward_stack = []
        cur_node = self
        while cur_node is not start_node:
            action_prev = cur_node.action_prev
            assert action_prev is not None
            reward = action_prev.reward
            cur_node = action_prev.state_prev
            assert cur_node is not None
            reward_stack.append(reward)

        i = 0
        cum_reward = 0
        while len(reward_stack) > 0:
            reward = reward_stack.pop()
            cum_reward += (discount**i)*reward
        return cum_reward

    def disentangle_paths(self):
        if len(self.action_next_list) == 0:
            state_cur_list = [deepcopy(self)]
        else:
            state_cur_list = []
            for action_next in self.action_next_list:
                state_next_list = action_next.state_next.disentangle_paths()
                for state_next in state_next_list:
                    state_cur = deepcopy(self)
                    action_edge = deepcopy(action_next)
                    action_edge.link_action2state(state_cur, state_next)
                    state_cur_list.append(state_cur)
        return state_cur_list

    def assign_v(self, discount):
        # unrolling the recursive function calls
        state_stack_order = [self]
        state_stack_compute = []
        while len(state_stack_order) != 0:
            state_popped = state_stack_order.pop()
            state_stack_compute.append(state_popped)
            for action_next in state_popped.action_next_list:
                state_stack_order.append(action_next.state_next)

        # recursively compute LB
        for state in state_stack_compute[::-1]:
            for action_next in state.action_next_list:
                v_max_next_state = action_next.state_next.v_search_tree
                q_max_cur_state = \
                    action_next.reward + discount*v_max_next_state
                state.v_search_tree = max(state.v_search_tree, q_max_cur_state)

    def get_natts2bds_ubd_unexhausted(self) -> Dict[str, List[Bidomain]]:
        natts2bds_unexhausted = defaultdict(list)
        # for each node attribute
        for natts, g2nids in self.natts2g2nids.items():
            left = g2nids['g1'] - self.exhausted_v
            right = g2nids['g2'] - self.exhausted_w
            if len(left) > 0 and len(right) > 0:
                natts2bds_unexhausted[natts].append(
                    Bidomain(left, right, None, natts)
                )
        return natts2bds_unexhausted

    def get_natts2bds_abd_unexhausted(self) -> Dict[str, List[Bidomain]]:
        natts2bds_unexhausted = defaultdict(list)
        for natts, bds in self.natts2bds.items():
            for bd in bds:
                assert bd.natts == natts
                if len(bd.left.intersection(self.exhausted_v)) > 0 or \
                        len(bd.right.intersection(self.exhausted_w)) > 0:
                    left = bd.left - self.exhausted_v
                    right = bd.right - self.exhausted_w
                    if len(left) > 0 and len(right) > 0:
                        natts2bds_unexhausted[natts].append(
                            Bidomain(left, right, None, natts)
                        )
                else:
                    assert len(bd.left) > 0 and len(bd.right) > 0
                    natts2bds_unexhausted[natts].append(bd)
        return natts2bds_unexhausted

    def get_natts2bds_unexhausted(self, with_bids=True) -> Dict[str, List[Bidomain]]:
        """
        Get a list of bidomains that are not exhausted, grouped by node attribute.
        :param with_bids: if True, (re)assign progressive ids to the bidomains (bids)
        """
        if len(self.nn_map) == 0:
            natts2bds_unexhausted = self.get_natts2bds_ubd_unexhausted()
        else:
            natts2bds_unexhausted = self.get_natts2bds_abd_unexhausted()
        if with_bids:
            assign_bids(natts2bds_unexhausted)
        return natts2bds_unexhausted

    def get_bd_idx(self, pruned_bidomains, v, w):
        for bd_idx, bidomain in enumerate(pruned_bidomains):
            if v in bidomain.left:
                assert w in bidomain.right
                return bd_idx
        return None

    def get_action_space_bds(self):
        # get pruned bidomains (pizza analogy)
        bds_unexhausted = self.get_unexhausted_bidomains()

        # get adjacent pruned bidomains
        bds_adjacent, bdids_adjacent = \
            self.get_adjacent_bidomains(bds_unexhausted)

        return bds_unexhausted, bds_adjacent, bdids_adjacent

    def get_adjacent_action_space_size(self):
        _, bds_adjacent, _ = self.get_action_space_bds()
        das = 0
        for bd in bds_adjacent:
            das += len(bd.left)*len(bd.right)
        return das

    def get_action_space_size(self, pruned_bds, selected_bds, num_nodes_max, Kprune, Lprune_l,
                              Lprune_r):
        pas = 0
        sas = 0
        for bd in pruned_bds:
            pas += len(bd.left)*len(bd.right)
        for bd in selected_bds:
            sas += len(bd.left)*len(bd.right)
        return pas, sas, len(pruned_bds), len(
            selected_bds), num_nodes_max, Kprune, Lprune_l + Lprune_r

    def log_action_space_size(self, pruned_bds, selected_bds, num_nodes_max, Kprune, Lprune_l,
                              Lprune_r, file_name):
        raw_line = self.get_action_space_size(pruned_bds, selected_bds, num_nodes_max, Kprune,
                                              Lprune_l,
                                              Lprune_r)
        line = ','.join([str(x) for x in raw_line]) + '\n'
        file = open(file_name, "a")
        file.write(line)
        file.close()

    def _filter_non_adj_bds(self, selected_bds, selected_bd_indices):
        if len(self.nn_map) == 0:
            return selected_bds, selected_bd_indices
        else:
            indices = []
            for idx, bd in enumerate(selected_bds):
                if bd.is_adj:
                    indices.append(idx)
            return [selected_bds[idx] for idx in indices], [selected_bd_indices[idx] for idx in
                                                            indices]

    def get_adjacent_bidomains(self, pruned_bidomains):
        if len(self.nn_map) > 0:
            adjacent_bds = [bd for bd in pruned_bidomains if bd.is_adj]
            adjacent_bdids = [i for i, bd in enumerate(pruned_bidomains) if bd.is_adj]
        else:
            adjacent_bds = pruned_bidomains
            adjacent_bdids = list(range(len(pruned_bidomains)))
        return adjacent_bds, adjacent_bdids

    def prune_action(self, v, w, remove_nodes=None):
        if not opt.scalable:
            self.pruned_actions.add_lr(v, w)
            if remove_nodes:
                for bidomain in self.bidomains:
                    if v in bidomain.left:
                        assert w in bidomain.right
                        if len(bidomain.right - self.pruned_actions.l2r[v]) == 0:
                            self.exhausted_v.add(v)
                            self.recompute = True
                        if len(bidomain.left - self.pruned_actions.r2l[w]) == 0:
                            self.exhausted_w.add(w)
                            self.recompute = True
        else:
            self.pruned_actions.add_lr(v, w)
            # if remove_nodes:
            for bd in unroll_bidomains(self.natts2bds):
                if v in bd.left:
                    assert w in bd.right
                    if len(bd.right - self.pruned_actions.l2r[v]) == 0:
                        self.exhausted_v.add(v)
                        self.recompute = True
                    if len(bd.left - self.pruned_actions.r2l[w]) == 0:
                        self.exhausted_w.add(w)
                        self.recompute = True

            for g2nids in self.natts2g2nids.values():
                if len(g2nids['g2'] - self.pruned_actions.l2r[v]) == 0:
                    self.exhausted_v.add(v)
                    self.recompute = True
                if len(g2nids['g1'] - self.pruned_actions.r2l[w]) == 0:
                    self.exhausted_w.add(w)
                    self.recompute = True

    def get_unexhausted_bidomains(self):
        bds_unexhausted = []
        for bd in self.bidomains:
            # if some of the nodes in this bidomain has been exhausted
            if len(bd.left.intersection(self.exhausted_v)) > 0 or \
                    len(bd.right.intersection(self.exhausted_w)) > 0:
                # form new bidomain excluding exhausted nodes
                bd_unexhausted = Bidomain(None, None, None, None)
                bd_unexhausted.left = bd.left - self.exhausted_v
                bd_unexhausted.right = bd.right - self.exhausted_w
                bd_unexhausted.is_adj = bd.is_adj
                if len(bd_unexhausted.left) > 0 and \
                        len(bd_unexhausted.right) > 0:
                    bds_unexhausted.append(bd_unexhausted)
            else:
                bds_unexhausted.append(bd)
        return bds_unexhausted


def assign_bids(natts2bds: Dict[str, List[Bidomain]]) -> None:
    """Assigns a progressive bid to each bidomain in natts2bds"""
    # to ensure that given the same natts2bds, bid assignment is deterministic => sorted
    bid = 0
    for natts in sorted(natts2bds.keys()):
        for bd in natts2bds[natts]:
            bd.bid = bid
            bid += 1
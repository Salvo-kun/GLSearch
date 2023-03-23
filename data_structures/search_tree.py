from __future__ import annotations
from .common import DoubleDict
from collections import defaultdict
from copy import deepcopy
import networkx as nx
from options import opt
from torch import Tensor
from data import *
from utils.mc_split import McspVec


def get_natts2g2abd_sg_nids(natts2g2nids: Dict[int, Dict[str, Set[int]]], natts2bds: dict, nn_map: dict) -> Dict[
    int, Dict[str, Set[int]]]:
    natts2g2abd_sg_nids = defaultdict(dict)
    sg1, sg2 = set(nn_map.keys()), set(nn_map.values())
    for natts, g2nid in natts2g2nids.items():
        left_cum, right_cum = set(), set()
        if natts in natts2bds:
            for bd in natts2bds[natts]:
                left_cum.update(bd.left)
                right_cum.update(bd.right)
        left_cum.update(sg1.intersection(g2nid['g1']))  # TODO: potential bottleneck O(nn_map)
        left_cum.update(sg2.intersection(g2nid['g2']))
        natts2g2abd_sg_nids[natts]['g1'] = left_cum
        natts2g2abd_sg_nids[natts]['g2'] = right_cum
    return natts2g2abd_sg_nids


#########################################################################
# Action Edge
#########################################################################
class ActionEdge(object):
    def __init__(self, action, q_vec_idx, reward,
                 pruned_actions=None, exhausted_v=None, exhausted_w=None):
        self.action = action
        self.q_vec_idx = q_vec_idx
        self.reward = reward

        # input to DQN
        self.pruned_actions = DoubleDict() \
            if pruned_actions is None else deepcopy(pruned_actions)
        self.exhausted_v = set() \
            if exhausted_v is None else deepcopy(exhausted_v)
        self.exhausted_w = set() \
            if exhausted_w is None else deepcopy(exhausted_w)

        # for search tree
        self.state_prev = None
        self.state_next = None

    def link_action2state(self, cur_state, next_state):
        self.state_prev = cur_state
        self.state_next = next_state
        cur_state.action_next_list.append(self)
        next_state.action_prev = self


#########################################################################
# Search Tree
#########################################################################
class SearchTree(object):
    def __init__(self, root: StateNode):
        self.root = root
        self.nodes = {root}
        self.edges = set()
        self.nxgraph = nx.Graph()

        # add root to nxgraph
        self.cur_nid = 0
        root.nid = self.cur_nid
        self.nxgraph.add_node(self.cur_nid)
        self.nid2ActionEdge = {}
        self.cur_nid += 1

        node_stack = [root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            for action_edge in node.action_next_list:
                node_next = action_edge.state_next
                node_next.cur_nid = self.cur_nid
                self.cur_nid += 1
                self.nodes.add(node_next)
                assert action_edge not in self.edges
                self.edges.add(action_edge)
                self.nxgraph.add_node(node_next.cur_nid)
                self.nid2ActionEdge[node_next.cur_nid] = action_edge
                self.nxgraph.add_edge(node.nid, node_next.cur_nid)
                node_stack.append(node_next)

    def link_states(self, cur_state, action_edge, next_state, q_pred, discount):
        action_edge.link_action2state(cur_state, next_state)

        assert cur_state in self.nodes
        self.nodes.add(next_state)
        assert action_edge not in self.edges
        self.edges.add(action_edge)

        # add edge, node to nxgraph
        assert cur_state.nid is not None
        next_state.nid = self.cur_nid
        self.nxgraph.add_node(self.cur_nid)
        self.nid2ActionEdge[self.cur_nid] = action_edge
        self.nxgraph.add_edge(cur_state.nid, self.cur_nid)
        eid = (cur_state.nid, self.cur_nid)
        self.assign_val_to_edge(eid, 'q_pred', q_pred.item())

        # accumulate the reward
        next_state.cum_reward = \
            self.get_next_cum_reward(cur_state, action_edge, discount)
        self.cur_nid += 1

    def get_next_cum_reward(self, cur_state, action_edge, discount):
        next_cum_reward = \
            cur_state.cum_reward + (discount**cur_state.num_steps)*action_edge.reward
        return next_cum_reward

    def disentangle_paths(self):
        search_tree_list = []
        root_list = self.root.disentangle_paths()
        for root in root_list:
            search_tree_list.append(SearchTree(root))
        return search_tree_list

    def assign_v_search_tree(self, discount):
        self.root.assign_v(discount)
        if not opt.scalable:
            for node in self.nodes:
                self.assign_val_to_node(node.nid, 'v_search_tree', node.v_search_tree)

    def associate_q_pred_true_with_node(self):
        for edge in self.edges:
            nid = edge.state_next.nid
            eid = (edge.state_prev.nid, edge.state_next.nid)
            q_pred = self.nxgraph.edges[eid]['q_pred']
            if 'q_true' in self.nxgraph.edges[eid]:
                q_true = self.nxgraph.edges[eid]['q_true']
            else:
                q_true = None
            if 'q_pred_norm' in self.nxgraph.edges[eid]:
                q_pred_norm = self.nxgraph.edges[eid]['q_pred_norm']
            else:
                q_pred_norm = None
            self.assign_val_to_node(nid, 'q_pred', q_pred)
            self.assign_val_to_node(nid, 'q_pred_str', '{:.2f}'.format(q_pred))
            if q_true is not None:
                self.assign_val_to_node(nid, 'q_true', q_true)
            if q_pred_norm is not None:
                self.assign_val_to_node(nid, 'q_pred_norm', q_pred_norm)
                self.assign_val_to_node(nid, 'q_pred_norm_str', '{:.3f}'.format(q_pred_norm))

    def assign_val_to_node(self, nid, key, val):
        self.nxgraph.nodes[nid][key] = val

    def assign_val_to_edge(self, eid, key, val):
        self.nxgraph.edges[eid][key] = val

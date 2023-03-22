from collections import defaultdict
from torch_geometric.utils import to_undirected
import torch
from options import opt
from data import *

def create_edge_index(g:Graph)->torch.Tensor:
    """
    Compute the edge index of a graph.
    :param g: A graph of type Graph.
    :return: A tensor of shape (2, 2*num_edges). Each colum is an edge. Each edge is represented twice (bot directions).
    """
    edge_index = torch.tensor(list(g.edges), device=opt.device).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=g.number_of_nodes())
    return edge_index

def create_adj_set(g:Graph) -> dict:
    adj_list = defaultdict(set)
    eit = torch.t(create_edge_index(g))
    for edge_index, edge in enumerate(eit):
        v, w = edge[0].item(), edge[1].item()
        adj_list[v].add(edge_index)
        adj_list[w].add(edge_index)
    return adj_list
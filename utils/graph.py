from collections import defaultdict
from torch_geometric.utils import to_undirected
import torch

def create_edge_index(g, device):
    edge_index = torch.tensor(list(g.edges), device=device).t().contiguous()
    edge_index = to_undirected(edge_index, num_nodes=g.number_of_nodes())
    return edge_index

def create_adj_set(g):
    adj_list = defaultdict(set)
    eit = torch.t(create_edge_index(g))
    for k, e in enumerate(eit):
        v, w = e[0].item(), e[1].item()
        adj_list[v].add(k)
        adj_list[w].add(k)
    return adj_list
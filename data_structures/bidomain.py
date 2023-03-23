from __future__ import annotations
from options import opt

class Bidomain(object):
    def __init__(self, left, right, is_adj, natts, bid=None):
        self.left = left
        self.right = right
        self.is_adj = is_adj
        self.natts = natts
        self.bid = bid

    def __len__(self):
        return len(self.left)*len(self.right)


def get_natts_hash(node: Dict[str, int]):
    if 'fuzzy_matching' in opt.reward_calculator_mode:
        natts = []
    else:
        natts = opt.node_feats_for_mcs
    natts_hash = tuple([node[natt] for natt in natts])
    return natts_hash


def unroll_bidomains(natts2bds: dict) -> list:
    bidomains = [bd for bds in natts2bds.values() for bd in bds]
    return bidomains
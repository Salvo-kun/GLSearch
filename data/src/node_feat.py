from torch_geometric.transforms import LocalDegreeProfile

# FIXME (maybe): removed dataset and one_hot encoder, check if it's actually needed
def encode_node_features(pyg_single_g):
    assert pyg_single_g is not None
    input_dim = pyg_single_g.x.shape[1] + 5
    pyg_single_g = LocalDegreeProfile()(pyg_single_g)
    
    if input_dim <= 0:
        raise ValueError('Must have at least one node feature encoder so that input_dim > 0')

    return pyg_single_g, input_dim






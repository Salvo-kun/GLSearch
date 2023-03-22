from __future__ import annotations
from models.base_model import BaseModel, ModelParams
from torch.nn import ModuleList
from models.mcs_rl_backtrack import MCSplitRLBacktrack
from models.mcs_rl_backtrack_scalable import MCSplitRLBacktrackScalable
from utils.options import extract_layer_info, get_option_value
from options import opt
from time import time


class GLSearch(BaseModel):
    def __init__(self, num_node_feat: int, feat_map: dict):
        super(GLSearch, self).__init__()
        self.scalable: bool = opt.scalable
        self.num_node_feat: int = num_node_feat
        self.feat_map: dict = feat_map
        self._init_layers()

    def forward(self, x: ModelParams) -> int:
        t = time()
        acts = [x.batch_data.merge_data['merge'].x]

        for layer in self.layers:
            ins = acts[-1]
            outs = layer(ModelParams.forge_layer_params(x, ins))
            acts.append(outs)

        total_loss = acts[-1]

        if not self.training:
            for pair in x.batch_data.pair_list:
                # Divide by the batch size and the running time is not precisely per-pair based.
                pair.assign_pred_time((time() - t)*1000/opt.batch_size)  # msec

        return total_loss

    def _init_layers(self) -> None:
        """
        Create the model layers and save them in GLSearch.layers
        """
        layers = ModuleList()

        for i in range(opt.layer_num):
            layer_name, layer_info = extract_layer_info(get_option_value(opt, f'layer_{i + 1}'))

            assert (layer_name == 'MCSRL_backtrack')  # only one supported, TODO move check in the option parser

            layer_info['in_dim'] = self.num_node_feat
            layer_info['feat_map'] = self.feat_map

            layers.append(
                MCSplitRLBacktrackScalable(**layer_info) if self.scalable else MCSplitRLBacktrack(**layer_info))

        self.layers = layers


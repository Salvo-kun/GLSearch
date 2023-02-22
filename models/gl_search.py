from models.base_model import BaseModel
from torch.nn import ModuleList
from models.mcs_rl_backtrack import MCSplitRLBacktrack
from models.mcs_rl_backtrack_scalable import MCSplitRLBacktrackScalable
from utils.options import extract_layer_info, get_option_value, parse_bool
import time

class GLSearch(BaseModel):
    def __init__(self, opt, num_node_feat, feat_map, tot_num_train_pairs):
        super(GLSearch, self).__init__(opt)
        self.scalable = parse_bool(get_option_value(self.opt, 'scalable'))
        self.num_node_feat = num_node_feat
        self.feat_map = feat_map
        self.tot_num_train_pairs = tot_num_train_pairs
        self._init_layers()

    def forward(self, cur_id, iter, batch_data): # TODO: rewrite this in the classic way (just x passed)
        t = time()
        acts = [batch_data.merge_data['merge'].x]
        
        for layer in self.layers:
            ins = acts[-1]
            outs = layer(ins, batch_data, iter=iter, cur_id=cur_id)
            acts.append(outs)
            
        total_loss = acts[-1]

        if not self.training:
            for pair in batch_data.pair_list:
                # Divide by the batch size and the running time is not precisely per-pair based.
                pair.assign_pred_time((time() - t) * 1000 / get_option_value(self.opt, 'batch_size'))  # msec
                
        return total_loss

    def _init_layers(self):
        layers = ModuleList()
        
        for i in range(get_option_value(self.opt, 'layer_num')):
            layer_name, layer_info = extract_layer_info(get_option_value(self.opt, f'layer_{i+1}'))
            
            assert(layer_name == 'MCSRL_backtrack') # only one supported, TODO move check in the option parser
            
            layer_info['num_node_feat'] = self.num_node_feat
            layer_info['feat_map'] = self.feat_map
            layer_info['tot_num_train_pairs'] = self.tot_num_train_pairs
            
            layers.append(MCSplitRLBacktrackScalable(self.opt, **layer_info) if self.scalable else MCSplitRLBacktrack(self.opt, **layer_info))
            
        self.layers = layers
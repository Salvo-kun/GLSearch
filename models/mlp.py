from torch.nn import Module, ModuleList, Linear, BatchNorm1d
from torch.nn.init import xavier_normal_, calculate_gain
from utils.layers import create_act

class MLP(Module):
    '''mlp can specify number of hidden layers and hidden layer channels'''

    def __init__(self, input_dim, output_dim, activation_type='relu', num_hidden_lyr=2, hidden_channels=None, bn=False):
        super().__init__()
        self.out_dim = output_dim
        
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
            
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError("number of hidden layers should be the same as the lengh of hidden_channels")
        
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)
        self.layers = ModuleList(list(map(self.weight_init, [Linear(self.layer_channels[i], self.layer_channels[i + 1]) for i in range(len(self.layer_channels) - 1)])))
        self.bn = BatchNorm1d(output_dim) if bn else None

    def weight_init(self, m):
        xavier_normal_(m.weight, gain=calculate_gain('relu'))
        return m

    def forward(self, x):
        layer_inputs = [x]
        for layer in self.layers:
            input = layer_inputs[-1]
            if layer == self.layers[-1]:
                layer_inputs.append(layer(input))
            else:
                layer_inputs.append(self.activation(layer(input)))
        # model.store_layer_output(self, layer_inputs[-1])
        if self.bn:
            layer_inputs[-1] = self.bn(layer_inputs[-1])
            
        return layer_inputs[-1]


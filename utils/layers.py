from torch.nn import ReLU, PReLU, Sigmoid, Tanh, ELU, Module

class Identity(Module):
    def forward(self, x):
        return x

def create_act(act:str , num_parameters=None):
    if act == 'relu' or act == 'ReLU':
        return ReLU()
    elif act == 'prelu':
        return PReLU(num_parameters)
    elif act == 'sigmoid':
        return Sigmoid()
    elif act == 'tanh':
        return Tanh()
    elif act == 'identity' or act == 'None':
        return Identity()
    if act == 'elu' or act == 'elu+1':
        return ELU()
    else:
        raise ValueError('Unknown activation function {}'.format(act))
PRETRAIN_MODE = 'pr'
IMITATION_MODE = 'il'
TRAIN_MODE = 'tr'
TEST_MODE = 'te'

class McspVec():
    def __init__(self, ldeg, rdeg):
        self.ldeg = ldeg
        self.rdeg = rdeg

class BufferEntry():
    def __init__(self, edge, g1, g2, search_tree):
        self.edge = edge
        self.g1 = g1
        self.g2 = g2
        self.search_tree = search_tree
        
class ForwardConfig():
    def __init__(self, total_runtime, recursion_threshold, q_signal, restore_bidomains, no_pruning, search_path, training):
        self.total_runtime = total_runtime
        self.recursion_threshold = recursion_threshold
        self.q_signal = q_signal
        self.restore_bidomains = restore_bidomains
        self.no_pruning = no_pruning
        self.search_path = search_path
        self.training = training

class MethodConfig():
    def __init__(self, DQN_mode, regret_iters):
        self.DQN_mode = DQN_mode
        self.regret_iters = regret_iters
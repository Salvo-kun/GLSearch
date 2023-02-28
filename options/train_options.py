from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument('--promise_mode', default='diverse')

    def parse(self):
        super().parse()
        
        self.opt.phase = 'train'
        
        return self.opt
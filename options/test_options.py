from options.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument('--promise_mode', default='P')

    def parse(self):
        super().parse()
        
        self.opt.phase = 'test'
        
        
        return self.opt

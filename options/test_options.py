from options.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

    def parse(self):
        super().parse()
        
        self.opt.phase = 'test'
        
        return self.opt

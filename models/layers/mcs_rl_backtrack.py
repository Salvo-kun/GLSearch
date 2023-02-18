from models.base_model import BaseModel

class MCSplitRLBacktrack(BaseModel):
    def __init__(self, opt, sample_size, **kwargs):
        super(MCSplitRLBacktrack, self).__init__(opt)

    def forward(self, x):
        pass

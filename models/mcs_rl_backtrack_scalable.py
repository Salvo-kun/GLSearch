from models.base_model import BaseModel

class MCSplitRLBacktrackScalable(BaseModel):
    def __init__(self, opt, sample_size, **kwargs):
        super(MCSplitRLBacktrackScalable, self).__init__(opt)

    def forward(self, x):
        pass

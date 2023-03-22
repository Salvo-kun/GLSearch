import torch
from torch.nn import Module
from abc import ABC, abstractmethod
from data.batch import BatchData

class ModelParams:
    def __init__(self,cur_id:int, iteration: int, batch_data: BatchData, ins=None):
        self.cur_id = cur_id
        self.iteration = iteration
        self.batch_data = batch_data
        self.ins = ins

    def forge_layer_params(self, ins: torch.Tensor):
        return ModelParams(self.cur_id, self.iteration, self.batch_data, ins)

class BaseModel(Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x:ModelParams) -> int:
        raise NotImplementedError




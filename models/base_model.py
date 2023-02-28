from torch.nn import Module
from abc import ABC, abstractmethod

class BaseModel(Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

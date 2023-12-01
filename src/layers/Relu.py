import torch

class Relu:
  def __init__(self, in_channels: int):
    self.in_channels = in_channels
  
  def apply(self, x: float):
    return 0 if x < 0 else x

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    return X.apply_(lambda x: self.apply(x))

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    return grad.apply_(lambda x: self.apply(x))
import torch

class Relu:
  def __init__(self, in_channels: int, device="cpu"):
    self.device = device
    self.in_channels = in_channels
  
  def apply(self, x):
    return torch.clamp(x, min=0)

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    return self.apply(X)

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    return self.apply(grad)
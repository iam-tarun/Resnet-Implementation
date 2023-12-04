import torch
# def Flatten(X: torch.Tensor) -> torch.Tensor:
#   return X.reshape(X.shape[0], -1)

class Flatten:
  def __init__(self, img_shape: [int, int, int]):
    self.img_shape = img_shape
  
  def forward(self, X: torch.Tensor):
    return X.reshape(X.shape[0], -1)
  
  def backward(self, grads: torch.Tensor):
    return grads.reshape(grads.shape[0], self.img_shape[0], self.img_shape[1], self.img_shape[2])
import torch

class softmax:
  def __init__(self):
    self.pla = None

  def forward(self,x):
    exp_x = torch.exp(x - torch.max(x)) 
    return exp_x / torch.sum(exp_x)

  def backward(self,x):
    return x * (1 - x)
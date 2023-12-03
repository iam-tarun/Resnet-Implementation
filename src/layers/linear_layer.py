import torch

class Linear:
  def __init__(self, in_features: int, out_features: int, device="cpu"):
    self.device = device
    self.in_features = in_features
    self.out_features = out_features
    self.weight = torch.randn(self.out_features, self.in_features, device=self.device)
    self.bias = torch.randn(self.out_features, device=self.device)
    self.input = None

  # shape of X should be (batch_size, n_features)
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    assert len(X.shape) == 2
    self.input = X
    return torch.matmul(X, self.weight.T) + self.bias.expand(X.shape[0], self.out_features)
  
  def backward(self, grad: torch.Tensor, lr: float = 0.01):
    assert self.input != None
    assert self.out_features == grad.shape[1]
    assert self.input.shape[0] == grad.shape[0]
    
    grad_weight = torch.matmul(grad.T, self.input)
    grad_bias = torch.sum(grad, dim=0)


    self.weight -= lr * grad_weight
    self.bias -= lr * grad_bias

    return torch.matmul(grad, self.weight)
  

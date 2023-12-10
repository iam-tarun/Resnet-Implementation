import torch

class BatchNorm:
  def __init__(self, n_channels: int, epsilon = 1e-5, momentum=0.9, device="cpu"):
    self.n_features = n_channels
    self.device = device
    self.epsilon = epsilon
    self.momentum = momentum
    self.weight = torch.randn(1, self.n_features, 1, 1, device=self.device)
    self.bias = torch.zeros(1, self.n_features, 1, 1, device=self.device)
    self.running_mean = torch.randn(1, self.n_features, 1, 1, device=self.device)
    self.running_var = torch.zeros(1, self.n_features, 1, 1, device=self.device)

    self.batch_size = None
    self.x_centered = None
    self.var_denom = None
    self.x_norm = None
  
  # shape (batch_size, n_channels, height, width)
  def forward(self, X: torch.Tensor, isTraining = True) -> torch.Tensor:
    if isTraining:

      batch_mean = torch.mean(X, dim=(0, 2, 3), keepdim=True)
      batch_var = torch.var(X, dim=(0, 2, 3), keepdim=True)

      # Update running mean and variance
      self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
      self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
      
      # Save variables for backpropagation
      self.batch_size = X.shape[0]
      self.x_centered = X - batch_mean
      self.var_denom = torch.sqrt(batch_var + self.epsilon)

      self.x_norm = self.x_centered / self.var_denom

      return (self.weight * self.x_norm) + self.bias
    
    else:
      self.x_norm = (X - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
      return self.weight * self.x_norm + self.bias

  def backward(self, grad: torch.Tensor, lr=0.01):
    # Calculate gradients for weight and bias
    b_grad = grad.sum(dim=(2, 3), keepdim=True)
    w_grad = torch.sum(grad * self.x_norm, dim=(2,3), keepdim=True)
    grad_x_norm = grad * self.weight
    grad_mean = -torch.sum(grad_x_norm, dim=0, keepdim=True) / torch.sqrt(self.running_var + self.epsilon)
    grad_var = -0.5 * torch.sum(grad_x_norm * (self.x_norm - self.running_mean),
                                    dim=0, keepdim=True) / torch.sqrt((self.running_var + self.epsilon)**3)
    
    grad_x = grad_x_norm / torch.sqrt(self.running_var + self.epsilon) + \
                 2 * grad_var * (self.x_norm - self.running_mean) / self.batch_size + \
                 grad_mean / self.batch_size
    # Update learnable parameters
    self.weight -= lr * w_grad.sum(dim=0)
    self.bias -= lr * b_grad.sum(dim=0)

    return grad_x

# input = torch.tensor([[[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]]], dtype=torch.float)
# grad = torch.tensor([[[[1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1]], [[1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1]]]], dtype=torch.float)
# input2 = input.clone()
# input2.requires_grad_(True)
# bn2 = torch.nn.BatchNorm2d(2, eps=1e-5, momentum=0.9)
# weight = torch.ones(2, dtype=torch.float, requires_grad=True)
# bias = torch.zeros(2, dtype=torch.float, requires_grad=True)
# bn2.weight.data = weight
# bn2.bias.data = bias
# bn1 = BatchNorm(2)
# x1 = bn1.forward(input)
# x2 = bn2(input2)
# # print(input)
# # print(input2)
# x2.backward(grad)
# print(input2.grad)
# x3 = bn1.backward(grad)
# print(x3)

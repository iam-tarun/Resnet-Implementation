import torch

class BatchNorm:
  def __init__(self, n_channels: int, epsilon = 1e-5, momentum=0.9, device="cpu"):
    self.n_features = n_channels
    self.device = device
    self.epsilon = epsilon
    self.momentum = momentum
    self.weight = torch.ones(1, self.n_features, 1, 1, device=self.device)
    self.bias = torch.zeros(1, self.n_features, 1, 1, device=self.device)
    self.running_mean = torch.ones(1, self.n_features, 1, 1, device=self.device)
    self.running_var = torch.zeros(1, self.n_features, 1, 1, device=self.device)

    self.batch_size = None
    self.x_centered = None
    self.var_denom = None
  
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

      X_normalized = self.x_centered / self.var_denom

      return (self.weight * X_normalized) + self.bias
    
    else:
      x_normalized = (X - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
      return self.weight * x_normalized + self.bias

  def backward(self, grad: torch.Tensor, lr=0.01):
    # Calculate gradients for weight and bias
    grad_weight = torch.sum(grad * self.x_centered / self.var_denom, axis=(0, 2, 3), keepdims=True)
    grad_bias = torch.sum(grad, axis=(0, 2, 3), keepdims=True)

    # Calculate gradient for x_normalized
    grad_x_normalized = grad * self.weight

    # Calculate gradients for x_centered and var_denom
    grad_x_centered = grad_x_normalized / self.var_denom
    grad_var_denom = torch.sum(grad_x_normalized * self.x_centered, axis=(0, 2, 3), keepdims=True)

    # Calculate gradients for batch mean and variance
    grad_batch_var = -0.5 * torch.sum(grad_var_denom * self.var_denom**(-3), axis=(0, 2, 3), keepdims=True)
    grad_batch_mean = -torch.sum(grad_x_centered / self.var_denom, axis=(0, 2, 3), keepdims=True) - \
                      2 * grad_batch_var * torch.mean(self.x_centered, axis=(0, 2, 3), keepdims=True)

    # Calculate gradient for x
    grad_x = grad_x_centered / self.var_denom + 2 * grad_batch_var * self.x_centered / self.batch_size + \
              grad_batch_mean / self.batch_size

    # Update parameters
    self.weight -= lr * grad_weight
    self.bias -= lr * grad_bias

    return grad_x

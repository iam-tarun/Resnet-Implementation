import torch

class AvgPool:
  def __init__ (self, kernel_size=2, device="cpu"):
    self.kernel_size = kernel_size
    self.device = device
  
  def forward(self, X: torch.Tensor):
    out_size = [X.shape[0], X.shape[1], X.shape[2]//self.kernel_size, X.shape[3]//self.kernel_size]
    y = torch.zeros(out_size, device=self.device)
    for r in range(0, y.shape[2]):
      for c in range(0, y.shape[3]):
        y[:, :, r, c] = torch.mean(X[:, :, r*self.kernel_size: r*self.kernel_size+self.kernel_size, c*self.kernel_size: c*self.kernel_size+self.kernel_size])
    return y
  
  def backward(self, grads: torch.Tensor):
    input_grads_size = [grads.shape[0], grads.shape[1], grads.shape[2]*self.kernel_size, grads.shape[3]*self.kernel_size]
    input_grads = torch.zeros(input_grads_size, device=self.device)
    for r in range(grads.shape[2]):
      for c in range(grads.shape[3]):
        d_avg = grads[:, :, r, c]/(self.kernel_size**2)
        input_grads[:, :, r*self.kernel_size: r*self.kernel_size+self.kernel_size, c*self.kernel_size:c*self.kernel_size+self.kernel_size] += torch.ones(grads.shape[0], grads.shape[1], self.kernel_size, self.kernel_size, device=self.device) * d_avg.view(grads.shape[0], grads.shape[1], 1, 1)
    
    return input_grads
  
# input = torch.tensor([[[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]]], dtype=torch.float)
# input2 = input.clone()
# input2.requires_grad_(True)
# avgp = AvgPool()
# x = avgp.forward(input)
# print(x)

# x2 = torch.nn.functional.avg_pool2d(input2, kernel_size=2, stride=2)
# print(x2)

# grads = torch.tensor([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=torch.float)
# x2.backward(grads)
# y = avgp.backward(grads)
# print(input2.grad)
# print(y)

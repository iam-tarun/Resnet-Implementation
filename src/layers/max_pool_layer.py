import torch

class MaxPool:
  def __init__(self, kernel_size= [2, 2], strides= 2, padding= 0, device="cpu"):
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.device = device
    self.input = None
  
  def forward(self, X: torch.Tensor) -> torch.Tensor:
    
    y = torch.zeros(X.shape[0], X.shape[1], int((X.shape[2] - self.kernel_size[0] + (2 * self.padding))/self.strides) + 1, int((X.shape[3] - self.kernel_size[1] + (2 * self.padding))/self.strides) + 1, device=self.device)
    if self.padding > 0:
      new_size = (X.shape[0], X.shape[1], X.shape[2] + 2*self.padding, X.shape[3] + 2*self.padding)
      x_padded = torch.zeros(new_size).to(device=self.device)
      x_padded[:, :, self.padding: x_padded.shape[2]-self.padding, self.padding: x_padded.shape[3]-self.padding] = X
      X = x_padded
    self.input = X
    for c in range(0, y.shape[3]):
      col_start = c * self.strides
      col_end = c * self.strides + self.kernel_size[1]
      for r in range(0, y.shape[2]):
        row_start = r * self.strides
        row_end = r * self.strides + self.kernel_size[0]
        y[:, :, r, c] = (torch.max((X[:, :, row_start: row_end, col_start: col_end]).reshape(X.shape[0], X.shape[1], 1, -1), dim=3)).values.reshape(X.shape[0], -1)
    
    return y

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    assert self.input != None
    grad_input_tensor = torch.zeros_like(self.input)
    
    for c in range(0, grad.shape[3]):
      col_start = c * self.strides
      col_end = c * self.strides + self.kernel_size[1]
      for r in range(0, grad.shape[2]):
        row_start = r * self.strides
        row_end = r * self.strides + self.kernel_size[0]

        input_slice = self.input[:, :, row_start: row_end, col_start: col_end]
        max_positions = torch.argmax(input_slice.reshape(self.input.shape[0], self.input.shape[1], -1), dim=2)
        
        max_i = max_positions // self.kernel_size[0]
        max_j = max_positions % self.kernel_size[0]

        grad_input_tensor[:, :, row_start: row_end, col_start: col_end][:, :, max_i, max_j] += grad[:, :, r, c]
    
    return grad_input_tensor
  
# input = torch.tensor([[[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]]], dtype=torch.float)
# input2 = input.clone()
# input2.requires_grad_(True)
# mp = MaxPool()
# x = mp.forward(input)
# # print(x)

# x2 = torch.nn.functional.max_pool2d(input2, kernel_size=2, stride=2)
# # print(x2)

# grads = torch.tensor([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=torch.float)
# x2.backward(grads)
# y = mp.backward(grads)
# print(input2.grad)
# print(y)
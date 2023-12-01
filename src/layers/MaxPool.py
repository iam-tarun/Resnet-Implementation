import torch

class MaxPool:
  def __init__(self, kernel_size= [2, 2], strides= 1, padding= 0, device="cpu"):
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.device = device
  
  def forward(self, X):
    y = torch.zeros(X.shape[0], X.shape[1], int((X.shape[2] - self.kernel_size[0] + (2 * self.padding))/self.strides) + 1, int((X.shape[3] - self.kernel_size[1] + (2 * self.padding))/self.strides) + 1, device=self.device)
    if self.padding > 0:
      new_size = (X.shape[0], X.shape[1], X.shape[2] + 2*self.padding, X.shape[3] + 2*self.padding)
      x_padded = torch.zeros(new_size).to(device=self.device)
      x_padded[:, :, self.padding: x_padded.shape[2]-self.padding, self.padding: x_padded.shape[3]-self.padding] = X
      X = x_padded

    for c in range(0, y.shape[3], self.strides):
      col_start = c
      col_end = c + self.kernel_size[1]
      for r in range(0, y.shape[2], self.strides):
        row_start = r
        row_end = r + self.kernel_size[0]
        y[:, :, row_start, col_start] = (torch.max((X[:, :, row_start: row_end, col_start: col_end]).reshape(X.shape[0], X.shape[1], 1, -1), dim=3)).values.reshape(X.shape[0], -1)
    
    return y

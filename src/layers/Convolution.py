import torch

class ConvLayer:
  def __init__(self, in_channels: int, out_channels: int, bias: bool= True, kernel_size = [3, 3], strides: int = 1, padding: int = 0, device: str = 'cpu'):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.n_kernels: int = int(out_channels / in_channels)
    self.device = device
    self.filters = torch.randn(self.n_kernels, self.kernel_size[0], self.kernel_size[1], device=self.device, requires_grad=True)
    self.bias = torch.zeros(1, self.n_kernels, requires_grad=True, device=self.device)
    self.doBias = bias

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    assert len(X.shape) == 4
    y = torch.zeros(X.shape[0], self.out_channels, int((X.shape[2] - self.kernel_size[0] + (2*self.padding))/self.strides) + 1, int((X.shape[3] - self.kernel_size[1]  + (2*self.padding))/self.strides) + 1, device=self.device)
    if self.padding > 0:
      # x = torch.nn.functional.pad(input=x, pad=(0, padding, 0, padding, 0, 0, 0, 0), mode='constant', value=0)
      new_size = (X.shape[0], X.shape[1], X.shape[2] + 2*self.padding, X.shape[3] + 2*self.padding)
      x_padded = torch.zeros(new_size).to(device=self.device)
      x_padded[:, :, self.padding: x_padded.shape[2]-self.padding, self.padding: x_padded.shape[3]-self.padding] = X
      X = x_padded
    # go through each column
    for c in range(0, y.shape[3], self.strides):
      col_start = c
      col_end = c + self.kernel_size[1]
      # go through each row
      for r in range(0, y.shape[2], self.strides):
        row_start = r
        row_end = r + self.kernel_size[0]
        # do the dot product of each 2d matrix in the input with each kernel and add all the elements in the resultant 2d matrix (convolution operation) for the entire batch at the current position with all kernels
        y[:, :, row_start, col_start] = (torch.matmul(X[:, :, row_start: row_end, col_start: col_end], self.filters.expand(X.shape[0], -1, self.kernel_size[0], self.kernel_size[1])).sum(dim=(2,3)).view(-1, self.out_channels)).to(device=self.device)
        if self.doBias:
          y[:, :, row_start, col_start] = y[:, :, row_start, col_start].add(self.bias)

    return y



 
import torch

class ConvLayer:
  def __init__(self, in_channels: int, out_channels: int, bias: bool= True, kernel_size = [3, 3], strides: int = 1, padding: int = 0, device: str = 'cpu'):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.n_kernels = out_channels
    self.device = device
    if self.kernel_size == [1, 1] :
      self.filters = torch.ones(1,1, dtype=torch.float, device=self.device)
    else:
      self.filters = torch.randn(self.n_kernels, self.in_channels, self.kernel_size[0], self.kernel_size[1], device=self.device)
    self.bias = torch.zeros(1, self.n_kernels, device=self.device)
    self.doBias = bias
    self.input = None
    self.filter_grads = None
    self.bias_grads = None

  def forward(self, X: torch.Tensor) -> torch.Tensor:
    assert len(X.shape) == 4
    y = torch.zeros(X.shape[0], self.out_channels, int((X.shape[2] - self.kernel_size[0] + (2*self.padding))/self.strides) + 1, int((X.shape[3] - self.kernel_size[1]  + (2*self.padding))/self.strides) + 1, device=self.device)
    if self.padding > 0:
      # x = torch.nn.functional.pad(input=x, pad=(0, padding, 0, padding, 0, 0, 0, 0), mode='constant', value=0)
      new_size = (X.shape[0], X.shape[1], X.shape[2] + 2*self.padding, X.shape[3] + 2*self.padding)
      x_padded = torch.zeros(new_size).to(device=self.device)
      x_padded[:, :, self.padding: x_padded.shape[2]-self.padding, self.padding: x_padded.shape[3]-self.padding] = X
      X = x_padded
    self.input = X
    # go through each column
    for im in range(X.shape[0]):
      for c in range(0, y.shape[3]):
        col_start = c * self.strides
        col_end = col_start + self.kernel_size[1]
        # go through each row
        for r in range(0, y.shape[2]):
          row_start = r * self.strides
          row_end = row_start + self.kernel_size[0]
          # do the dot product of each 2d matrix in the input with each kernel and add all the elements in the resultant 2d matrix (convolution operation) for the entire batch at the current position with all kernels
          y[im, :, r, c] = (X[im, :, row_start: row_end, col_start: col_end] * self.filters).sum(dim= (1,2,3) if self.kernel_size[0] != 1 else (1,2)).to(device=self.device)
          if self.doBias:
            y[im, :, r, c] = y[im, :, r, c].add(self.bias)

    return y

  
  # loss grad resolution is [(input_rows - filter_size + 2*padding)/strides + 1, (input_cols - filter_size + 2*padding)/strides + 1]
  # loss grad depth is equal to in_channels
  # for understanding lets take: strides = 1 and 0 padding
  # input dims = [2, 3, 8, 8]
  # filter dims = [6, 3, 3, 3]
  # output dims = [2, 6, 6, 6]
  # loss_grad dims = [2, 6, 6, 6]
  # filter grad dims = [6, 3, 3, 3]
  # bias dims = [1, 6]
  # bias grad dims = [1, 6] (sum over all values in dim 0, 1 of loss_grad)
  # input_grads dims == [2, 3, 8, 8] (all values over batch will be accumulated)
  # new rotated filters dim = [6, 3, 13, 13]
  def backward(self, loss_grad: torch.Tensor, lr=0.01) -> torch.Tensor:
    # calculating the gradient for filters
    self.filter_grads = torch.zeros_like(self.filters)
    if self.strides != 1:
      out_size = [self.input.shape[0], self.out_channels, self.input.shape[2] - self.kernel_size[0] + 1, self.input.shape[3] - self.kernel_size[1] + 1]
      new_grad = torch.zeros(out_size)

      for r in range(0, new_grad.shape[2], self.strides):
        for c in range(0, new_grad.shape[3], self.strides):
          new_grad[:, :, r,c] = loss_grad[:, :, r//self.strides, c//self.strides]

      loss_grad = new_grad
    
    for ch in range(self.filter_grads.shape[1]):
      for c in range(self.filter_grads.shape[3]):
        for r in range(self.filter_grads.shape[2]):
          row_start = r
          row_end = row_start + loss_grad.shape[2]
          col_start = c
          col_end = col_start + loss_grad.shape[3]
          # [6, 3] = ( [2, 3, 6, 6] * [2, 6, 6, 6] ) sum over 
          self.filter_grads[:, ch, row_start, col_start] = torch.sum(self.input[:, ch, row_start: row_end, col_start: col_end].unsqueeze(0).repeat(1, self.input.shape[0], 1, 1) * loss_grad, dim=(0, 2, 3))
    
    # calculating the gradients for bias
    self.bias_grads = torch.zeros_like(self.bias)
    self.bias_grads = loss_grad.sum(dim=(2,3))
   
    new_size = (loss_grad.shape[0], loss_grad.shape[1], loss_grad.shape[2] + 2*(self.kernel_size[0]-1), loss_grad.shape[3] + 2*(self.kernel_size[1]-1))
    padd_grad = torch.zeros(new_size)
    padd_grad[:, :, self.kernel_size[0]-1: loss_grad.shape[2] + self.kernel_size[0]-1, self.kernel_size[1]-1: loss_grad.shape[3] + self.kernel_size[1]-1] = loss_grad
    loss_grad = padd_grad
    
    # calculating the input loss gradients to return
    input_grads = torch.zeros_like(self.input)
    rotated_filters = torch.flip(self.filters, dims=(2, 3))
    # new_rotated_filter_size = (rotated_filters.shape[0], rotated_filters.shape[1],  2*(loss_grad.shape[2]-1) + self.kernel_size[0], 2*(loss_grad.shape[3]-1) + self.kernel_size[1])
    # rotated_filters_padded = torch.zeros(new_rotated_filter_size).to(device=self.device)
    # rotated_filters_padded[:, :, 
    # loss_grad.shape[2] - 1: loss_grad.shape[2] - 1 + self.kernel_size[0], loss_grad.shape[3] - 1: loss_grad.shape[3] - 1 + self.kernel_size[1]] = rotated_filters
    
    for ch in range(self.input.shape[1]):
      for r in range(self.input.shape[2]):
        for c in range(self.input.shape[3]):
          # [1, 3] = ([6, 3, 6, 6] * [2, 6, 6, 6]) sum on dims 0,2 and 3
          input_grads[:, ch, r, c] = torch.sum(rotated_filters[:, ch, :, :] * loss_grad[:, :, r:r+self.kernel_size[0], c:c+self.kernel_size[1]], dim=(1,2,3))

    self.filters -= lr*self.filter_grads
    self.bias -= lr * self.bias_grads.sum(dim=(0))
    if self.padding:
      input_grads = input_grads[:, :, self.padding: self.input.shape[2] - self.padding, self.padding: self.input.shape[3] - self.padding]
    return input_grads


# # pytorch lib method
# c2 = ConvLayer(2, 4)
# input = torch.tensor([[[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]],
#                       [[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]]
#                       ], dtype=torch.float)
# input2 = input.clone()
# input2.requires_grad_(True)
# o2 = c2.forward(input)
# print(o2.shape)
# grads = torch.tensor([[[[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]]], [[[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=torch.float)
# # grads = torch.tensor([[[[1]], [[0]], [[1]], [[0]]]], dtype=torch.float)
# # grads = torch.tensor([[[[1, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]]]], dtype=torch.float)
# # print(grads.shape)
# g3 = c2.backward(grads)
# print(o2)

# import torch.nn.functional as F
# custom_kernels = torch.ones(4, 2, 3, 3, requires_grad=True)
# custom_gradients = grads
# bias_val = torch.zeros(4, requires_grad=True)
# o1 = F.conv2d(input2, custom_kernels, bias=bias_val)
# print(o1)
# o1.backward(grads)
# # print(custom_kernels.grad)
# # print(c2.filter_grads)
# # print(bias_val.grad)
# # print(c2.bias_grads)
# print(input2.grad)
# print(g3)
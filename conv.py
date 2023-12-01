import torch
import torch.nn.functional as F

# Create a 3x3 matrix (input)
input_matrix = torch.tensor([[[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]],
                              [[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]]], dtype=torch.float32)

# Create a 3x3 kernel
kernel = torch.tensor([[[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]],
                      [[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]]], dtype=torch.float32)

# Add batch and channel dimensions to the input matrix (required for conv2d)
input_matrix = input_matrix.view(2, 1, 3, 3)

# Add batch and channel dimensions to the kernel (required for conv2d)

bias = torch.randn(1, 2)
print(bias)
res = (torch.matmul(input_matrix, kernel.expand(2, -1, 3, 3))).sum(dim=(2, 3)).add(bias)
print(res)

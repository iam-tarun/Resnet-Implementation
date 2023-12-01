import torch
import torch.nn as nn

class SimpleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SimpleConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        batch_size, _, input_height, input_width = x.size()
        output_height = ((input_height + 2 * self.padding - self.kernel_size) // self.stride) + 1
        output_width = ((input_width + 2 * self.padding - self.kernel_size) // self.stride) + 1
        x_padded = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x_reshaped = x_padded.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x_reshaped = x_reshaped.contiguous().view(batch_size, self.in_channels, -1, self.kernel_size, self.kernel_size)
        weight_reshaped = self.weight.unsqueeze(2).unsqueeze(3)
        output = torch.sum(x_reshaped * weight_reshaped, dim=(4, 5))
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        output = output.view(batch_size, self.out_channels, output_height, output_width)
        return output
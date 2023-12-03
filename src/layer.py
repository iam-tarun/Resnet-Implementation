from layers.conv_layer import ConvLayer
from layers.batch_norm_layer import  BatchNorm
from layers.relu_layer import Relu
import torch

class ResLayer:
  def __init__(self, in_channels, out_channels, stride=1, skip_layer=None):
    self.conv1 = ConvLayer(in_channels, out_channels)
    self.bn1 = BatchNorm(out_channels)
    self.conv2 = ConvLayer(out_channels, out_channels, strides=stride, padding=2)
    self.bn2 = BatchNorm(out_channels)
    self.relu1 = Relu(out_channels)
    self.relu2 = Relu(out_channels)
    self.skip_layer = skip_layer
    self.input = None
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    self.input = x
    x = self.conv1.forward(x)
    x = self.bn1.forward(x)
    x = self.relu1.forward(x)
    x = self.conv2.forward(x)
    x = self.bn2.forward(x)

    if self.skip_layer is not None:
      self.input = self.skip_layer(self.input)
    
    # x += self.input
    x = self.relu2.forward(x)
    return x

  def backward(self, grad: torch.Tensor) -> torch.Tensor:
    grads = self.relu2.backward(grad)
    grads = self.bn2.backward(grads)
    grads = self.conv2.backward(grads)
    grads = self.relu1.backward(grads)
    grads = self.bn1.backward(grads)
    grads = self.conv1.backward(grads)

    return grads
  

# testing reslayer

input = torch.randn(1, 64, 56, 56, dtype=torch.float)
rl = ResLayer(64, 64, 2)
o = rl.forward(input)
print(o.shape)
grads = torch.randn(1, 64, 28, 28, dtype=torch.float)
g = rl.backward(grads)
print(g.shape)

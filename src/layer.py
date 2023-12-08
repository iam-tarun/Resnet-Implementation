from layers.conv_layer import ConvLayer
from layers.batch_norm_layer import  BatchNorm
from layers.relu_layer import Relu
import torch

class ResBlock:
  def __init__(self, in_channels, out_channels, stride=1, skip_layer=None, device="cpu"):
    self.device = device
    self.conv1 = ConvLayer(in_channels, out_channels, device=self.device)
    self.bn1 = BatchNorm(out_channels, device=self.device)
    self.conv2 = ConvLayer(out_channels, out_channels, strides=stride, padding=2, device=self.device)
    self.bn2 = BatchNorm(out_channels, device=self.device)
    self.relu1 = Relu(out_channels, device=self.device)
    self.relu2 = Relu(out_channels, device=self.device)
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
      self.input = self.skip_layer.forward(self.input)
    
    x += self.input
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
# input = torch.randn(1, 64, 56, 56, dtype=torch.float)
# rl = ResBlock(64, 64, 2)
# o = rl.forward(input)
# print(o.shape)
# grads = torch.randn(1, 64, 28, 28, dtype=torch.float)
# g = rl.backward(grads)
# print(g.shape)

class ResLayer:
  def __init__(self, in_channels: int, out_channels: int, skip_layer: bool = True, device="cpu"):
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.device = device
    self.identity_conv = ConvLayer(in_channels, out_channels, kernel_size=[1, 1], strides=2, device= self.device)
    self.res_block1 = ResBlock(in_channels, out_channels, 2, self.identity_conv if skip_layer else None, device=self.device)
    self.res_block2 = ResBlock(out_channels, out_channels, device=self.device)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.res_block1.forward(x)
    x = self.res_block2.forward(x)
    return x

  def backward(self, grads: torch.Tensor) -> torch.Tensor:
    grad = self.res_block2.backward(grads)
    grad = self.res_block1.backward(grad)
    return grad
  
# testing reslayer
# from PIL import Image
# from torchvision import transforms
# import matplotlib.pyplot as plt
# import numpy as np

# transform = transforms.Compose([
#   transforms.ToTensor(),
# ])
# img = Image.open("src/lena.gif")
# img = np.array(img)
# img.resize(3, 56, 56)
# x = transform(img)
# x = x.unsqueeze(0)
# x = x.reshape(1, 3, 56, 56)
# print(x.shape)
# rl1 = ResLayer(3, 64)
# rl2 = ResLayer(64, 128)
# rl3 = ResLayer(128, 256)
# rl4 = ResLayer(256, 512)
# # x = torch.randn(1, 3, 56, 56)
# x = rl1.forward(x)
# print(x.shape)
# x = rl2.forward(x)
# print(x.shape)
# x = rl3.forward(x)
# print(x.shape)
# x = rl4.forward(x)
# print(x.shape)
# grad = torch.randn_like(x)
# grads = rl4.backward(grad)
# print(grads.shape)
# grads = rl3.backward(grads)
# print(grads.shape)
# grads = rl2.backward(grads)
# print(grads.shape)
# grads = rl1.backward(grads)
# print(grads.shape)
# plt.imshow(grads[0].permute(1, 2, 0).cpu().numpy())
# plt.show()
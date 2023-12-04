from layer import ResLayer
from layers.conv_layer import ConvLayer
from layers.max_pool_layer import MaxPool
from layers.avg_pool_layer import AvgPool
from layers.flattern_layer import Flatten
from layers.linear_layer import Linear
from layers.softmax_layer import softmax
import torch
from layers.cce import CategoricalCrossEntropyLoss

class ResNet:
  def __init__(self, img_channels: [int, int, int]):
    self.img_channels = img_channels
    self.conv1 = ConvLayer(img_channels[0], 64, strides=2, kernel_size=[7, 7])
    self.mPool = MaxPool(kernel_size=[3, 3], strides=2, padding=1)
    self.layer1 = ResLayer(64, 64)
    self.layer2 = ResLayer(64, 128)
    self.layer3 = ResLayer(128, 256)
    self.layer4 = ResLayer(256, 512)
    self.avgPool = AvgPool()
    self.flatten = Flatten([512, 2, 2])
    self.linear = Linear(2048, 67)
    self.softmax = softmax()
  
  def forward(self,  x: torch.Tensor):
    x = self.conv1.forward(x) # batch_size x 64 x 112 x 112
    x = self.mPool.forward(x) # batch_size x 64 x 56 x 56
    x = self.layer1.forward(x) # batch_size x 64 x 28 x 28
    x = self.layer2.forward(x) # batch_size x 128 x 14 x 14
    x = self.layer3.forward(x) # batch_size x 256 x 7 x 7
    x = self.layer4.forward(x) # batch_size x 512 x 4 x 4
    x = self.avgPool.forward(x) # batch_size x 512 x 2 x 2
    x = self.flatten.forward(x) # batch_size x 2048
    x = self.linear.forward(x) # batch_size x 67
    x = self.softmax.forward(x) # batch_size x 67
    return x
  
  def backward(self, grad: torch.Tensor):
    grads = self.softmax.backward(grad)
    grads = self.linear.backward(grads)
    grads = self.flatten.backward(grads)
    grads = self.avgPool.backward(grads)
    grads = self.layer4.backward(grads)
    grads = self.layer3.backward(grads)
    grads = self.layer2.backward(grads)
    grads = self.layer1.backward(grads)
    grads = self.mPool.backward(grads)
    self.conv1.backward(grads)


# testing
# rn = ResNet([3, 229, 229])
# x = torch.randn(1, 3, 229, 229)
# y = rn.forward(x)
# cce = CategoricalCrossEntropyLoss()
# y_true = torch.zeros(1, 67)
# y_true[0][2] = 1
# loss = cce.forward(y, y_true)
# grad = cce.backward(y, y_true)
# print(loss)
# rn.backward(grad)

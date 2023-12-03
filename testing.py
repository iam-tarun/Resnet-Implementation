# import torch
# from PIL import Image
# from torchvision import transforms
# img_path = "./sample.ppm"
# transform =  transforms.Compose([
#   transforms.Grayscale(),
#   transforms.ToTensor(),
# ])

# from src.layers.Convolution import ConvLayer
# from layers.batch_norm_layer import BatchNorm
# from src.layers.MaxPool import MaxPool
# from src.layers.Relu import Relu
# from src.layers.Flatten import Flatten
# from src.layers.Linear import Linear
# from layers.cce import CategoricalCrossEntropyLoss

# img = Image.open(img_path)
# tensor_image = transform(img)
# tensor_image = (tensor_image.reshape(1, tensor_image.shape[0], tensor_image.shape[1], tensor_image.shape[2]))
# print(tensor_image.shape) # [1, 1, 623, 625]

# conv1 = ConvLayer(1, 3 )
# bn1 = BatchNorm(3)

# conv2 = ConvLayer(3, 6)
# bn2 = BatchNorm(6)

# Mpool = MaxPool()
# relu = Relu(6)

# lin1 = Linear(574740, 10)
# lin2 = Linear(10, 1)
# #forward
# # initial shape [1, 1, 623, 625]
# x = conv1.forward(tensor_image)
# # print(x.shape)
# # [1, 3, 621, 623]
# x = bn1.forward(x)
# # print(x.shape)
# # [1, 3, 621, 623]
# x = conv2.forward(x)
# x = bn2.forward(x)
# x = Mpool.forward(x)
# x = relu.forward(x)
# x = Flatten(x)
# x = lin1.forward(x)
# x = lin2.forward(x)
# lossFn = CategoricalCrossEntropyLoss()
# loss = lossFn.forward(x, torch.Tensor([[1]]))
# # backward
# grad = lossFn.backward(x, torch.Tensor([[1]] ))
# grad = lin2.backward(grad)
# grad = lin1.backward(grad)
# grad = grad.reshape(1, 6, 309, 310)
# grad = relu.backward(grad)
# grad = Mpool.backward(grad)
# grad = bn2.backward(grad)
# grad = conv2.backward(grad)
# print(x.shape)

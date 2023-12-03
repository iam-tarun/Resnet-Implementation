import torch

class CategoricalCrossEntropyLoss:
  def __init__(self):
    pass

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Clip predicted values to avoid log(0) issues
    eps = 1e-5
    y_pred = torch.clip(y_pred, eps, 1 - eps)

    loss = -torch.sum(y_true * torch.log(y_pred)) / len(y_true)
    return loss
  
  def backward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    y_pred = torch.clip(y_pred, eps, 1 - eps)

    grad_loss = -y_true / (y_pred + eps)
    return grad_loss

# y_pred = torch.Tensor([
#     [0.6, 0.2, 0.2],
#     [0.2, 0.7, 0.1],
#     [0.1, 0.2, 0.7],
#     [0.8, 0.1, 0.1],
# ])

# y_true = torch.Tensor([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 0],
# ])

# cce = CategoricalCrossEntropyLoss()
# loss = cce.forward(y_pred, y_true)
# print(loss)
# grad = cce.backward(y_pred, y_true)
# print(grad)
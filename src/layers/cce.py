import torch

class CategoricalCrossEntropyLoss:
  def __init__(self, device="cpu"):
    self.y = None
    self.device = device

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # Clip predicted values to avoid log(0) issues
    eps = 1e-5
    y_pred = torch.clip(y_pred, eps, 1 - eps)
    self.y = torch.zeros_like(y_pred, device=self.device)
    for i in range(y_pred.shape[0]):
      self.y[i] = torch.tensor([int(bit) for bit in format(y_true[i], 'b').zfill(67)], dtype=torch.int)
    loss = -torch.sum(self.y * torch.log(y_pred)) / len(y_true)
    return loss
  
  def calculate_metrics(self, predictions, targets):
    # Convert predictions to binary (0 or 1) using a threshold, e.g., 0.5 for binary classification
    binary_predictions = (predictions > 0.5).float()
    y = torch.zeros_like(predictions, device=self.device)
    for i in range(predictions.shape[0]):
      self.y[i] = torch.tensor([int(bit) for bit in format(targets[i], 'b').zfill(67)], dtype=torch.int)
    # Convert to numpy arrays
    targets_np = y
    predictions_np = binary_predictions.cpu().numpy()

    # Calculate metrics
    accuracy = self.accuracy(targets_np, predictions_np)
    precision, recall, f1 = self.precision_recall_f1(targets_np, predictions_np, average='binary')
  
    return accuracy, precision, recall, f1

  def backward(self, y_pred: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    y_pred = torch.clip(y_pred, eps, 1 - eps)

    grad_loss = -self.y / (y_pred + eps)
    return grad_loss

  def accuracy(self, predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total
  
  def precision_recall_f1(self, predictions, labels, threshold=0.5):
    # Apply threshold to convert probability predictions to binary
    binary_predictions = (predictions >= threshold).float()

    true_positives = torch.sum((binary_predictions == 1) & (labels == 1)).item()
    false_positives = torch.sum((binary_predictions == 1) & (labels == 0)).item()
    false_negatives = torch.sum((binary_predictions == 0) & (labels == 1)).item()

    precision = true_positives / max((true_positives + false_positives), 1e-10)
    recall = true_positives / max((true_positives + false_negatives), 1e-10)
    
    f1 = 2 * (precision * recall) / max((precision + recall), 1e-10)

    return precision, recall, f1
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
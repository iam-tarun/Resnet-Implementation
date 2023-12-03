import torch
def Flatten(X: torch.Tensor) -> torch.Tensor:
  return X.reshape(X.shape[0], -1)


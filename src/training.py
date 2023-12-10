from utils.dataload import CustomDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np
from model import ResNet
from layers import cce
from utils.load_config import load_training_config
from utils.save_logs import save_metrics_to_log
config = load_training_config()


transform = transforms.Compose([
    transforms.Resize((229, 229)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomDataset(root_dir='Images', transform=transform)


# batch_size = 71
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print(len(dataset))
# print(batch_size)

# i=0
# # Example: Iterate through the dataset
# for batch_idx, (images, labels) in enumerate(data_loader):
#     print(images.shape)

dataset_size = len(dataset)

# Define the split ratios (e.g., 80% training, 20% testing)
train_ratio = 0.8
test_ratio = 1 - train_ratio

# Calculate the sizes of training and testing sets
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

# Use random_split to split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create separate DataLoaders for training and testing sets
batch_size = 71
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# for batch_idx, (images, labels) in enumerate(train_data_loader):
#      print(images.shape)

# for batch_idx, (images, labels) in enumerate(train_data_loader):
    # Convert the torch tensor to numpy array
    # images_np = images.numpy()

    # # Denormalize the images (if normalization was applied)
    # mean = np.array([0.5, 0.5, 0.5])  # Update with your normalization mean
    # std = np.array([0.5, 0.5, 0.5])   # Update with your normalization std
    # #images_np = images_np * std + mean

    # # Transpose the shape to (batch_size, height, width, channels)
    # images_np = np.transpose(images_np, (0, 2, 3, 1))

    # # Plot the first 10 images
    # num_images_to_plot = 10
    # for i in range(min(num_images_to_plot, len(images_np))):
    #     plt.subplot(2, 5, i + 1)
    #     plt.imshow(images_np[i])
    #     plt.axis('off')
    #     plt.title(f'Class: {labels[i]}')

    # plt.show()

#     if batch_idx == 0:  # Stop after the first batch
#         break


# epochs = 10

class TrainModel(torch.nn.Module):
    def __init__(self, img_channels, train_dataLoader, test_dataLoader, cur_epoch: int, n_epochs: int):
        super(TrainModel, self).__init__()
        self.resnet = ResNet(img_channels, config['device'])
        self.train_dataLoader = train_dataLoader
        self.test_dataLoader = test_dataLoader
        self.n_epochs = n_epochs
        self.epochs = cur_epoch
        self.cce = cce.CategoricalCrossEntropyLoss(device=config['device'])
    
    def fit(self):
        total_loss = 0
        predictions_list = []
        targets_list = []
        for _ in range(self.epochs, self.n_epochs):
            for __, (images, labels) in enumerate(self.train_dataLoader):
                y_pred = self.resnet.forward(images.to(device=config['device']))
                loss = self.cce.forward(y_pred, labels)
                total_loss += loss.item()
                predictions_list.append(y_pred)
                targets_list.append(labels)
                print(loss)
                grad = self.cce.backward(y_pred)
                self.resnet.backward(grad)
            predictions_tensor = torch.cat(predictions_list)
            targets_tensor = torch.cat(targets_list)
            accuracy, precision, recall, f1 = self.cce.calculate_metrics(predictions_tensor, targets_tensor)
            average_loss = total_loss / len(self.train_dataLoader)
            print(f'Epoch {_ + 1}: Loss={average_loss:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}')
            save_metrics_to_log(_, average_loss, accuracy, precision, recall, f1, config['logging']['log_file'])
            if (_ + 1) % config['save_interval'] == 0:
                checkpoint_path = config['model_save']['save_path'].format(_+1)
                torch.save(self.state_dict(), checkpoint_path)
                print(f'Model saved at epoch {_ + 1} to {checkpoint_path}')

m = TrainModel([3, 229, 229], train_data_loader, test_data_loader, 0, 10)
m.fit()

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.class_labels = sorted(os.listdir(root_dir))  # Get sorted class labels
        self.file_list = self.get_file_list()

    def get_file_list(self):
        file_list = []
        for class_label in self.class_labels:
            class_path = os.path.join(self.root_dir, class_label)
            files = [os.path.join(class_path, file) for file in os.listdir(class_path)]
            file_list.extend(files)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        # Extract label from the folder name
        class_label = os.path.basename(os.path.dirname(img_name))
        label = self.class_labels.index(class_label)

        return image, label

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((229, 229)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomDataset(root_dir='path_to_image', transform=transform)



'''
your_dataset_path
|-- class_label_1
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|-- class_label_2
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|-- ...

'''

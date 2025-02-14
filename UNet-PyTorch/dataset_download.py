import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Download and load the dataset
train_dataset = datasets.OxfordIIITPet(root='./data', download=True, transform=transform, target_transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
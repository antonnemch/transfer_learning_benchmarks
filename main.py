# train_kaggle_adapters.py
# Train ResNet-50 with ConvAdapters on Kaggle Brain MRI dataset

import torch
from torch import nn, optim
from models.resnet_base import resnet50_base, add_adapters_to_resnet, freeze_encoder
from utils.loaders import load_kaggle_brain_mri
from utils.train_resnet import train_one_epoch, count_parameters, count_parameters_by_module

# Config
kaggle_path = r"C:\Users\anton\Documents\Datasets\Kaggle Brain MRI"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
batch_size = 32
lr = 1e-3

# Load dataset
train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(kaggle_path, batch_size=batch_size)

# Initialize model
model = resnet50_base(pretrained=True, num_classes=num_classes)
add_adapters_to_resnet(model, reduction=4)
freeze_encoder(model)
model.to(device)

# Count parameters
count_parameters(model)
count_parameters_by_module(model)

# Optimizer (only trainable parameters)
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# Train loop
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, train_loader, optimizer, device)

# Optional: save the model
torch.save(model.state_dict(), "adapter_trained_resnet50.pth")

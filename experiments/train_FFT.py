# train_finetune.py
# Standard full fine-tuning of ResNet-50 on Kaggle Brain MRI

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models.resnet_base import resnet50_base
from utils.loaders import load_kaggle_brain_mri
from utils.train_resnet import count_parameters, count_parameters_by_module
from utils.dataset_summarize import summarize

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def initialize_model(num_classes, device):
    model = resnet50_base(pretrained=True, num_classes=num_classes, is_lora=False)
    model.to(device)
    return model

def train_finetune_on_kaggle_bmri(
    kaggle_path,
    batch_size=16,
    num_epochs=5,
    lr=1e-3,
    save_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(kaggle_path, batch_size=batch_size)

    # Initialize model
    model = initialize_model(num_classes, device)

    # Inspect parameters
    count_parameters(model)
    count_parameters_by_module(model)

    # Initialize optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        model.train()
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f}")

        val_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    filename = f"resnet50_finetune_bs{batch_size}_lr{lr}_ep{num_epochs}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    kaggle_path = r"C:\\Users\\anton\\Documents\\Datasets\\Kaggle Brain MRI"
    train_finetune_on_kaggle_bmri(kaggle_path)

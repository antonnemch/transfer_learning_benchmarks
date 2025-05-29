# train_kaggle_meta.py
# Train ResNet-50 with MetaLR on Kaggle Brain MRI dataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from models.resnet_base import resnet50_base
from models.meta50 import MetaSGD
from utils.loaders import load_kaggle_brain_mri
from utils.train_resnet import count_parameters, count_parameters_by_module
from utils.dataset_summarize import summarize
from utils.meta_train import get_optimizer, train_meta_step


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


def train_meta_on_kaggle_bmri(
    kaggle_path,
    batch_size=16,
    num_epochs=5,
    lr=1e-2,
    hyper_lr=1e-1,
    save_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(kaggle_path, batch_size=batch_size)
    # summarize("Kaggle Brain MRI", train_loader, val_loader, test_loader, num_classes)

    # Initialize models
    model = initialize_model(num_classes, device)
    meta_model = initialize_model(num_classes, device)

    train_meta_step.meta_iterator = iter(val_loader)

    # Inspect parameters
    count_parameters(model)
    count_parameters_by_module(model)

    # Initialize learning rates
    now_lr = [0.1 * lr] * 18 + [lr]  # 18 groups: 17 conv/fc blocks + 1 fc
    optimizer = get_optimizer(model, now_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        model.train()
        num_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            loss, _, now_lr = train_meta_step(
                images, labels, model, meta_model, now_lr, val_loader,
                optimizer, criterion, epoch,
                batch_idx=batch_idx, num_batches=num_batches
            )

        val_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    filename = f"resnet50_meta_bs{batch_size}_lr{lr}_hyperlr{hyper_lr}_ep{num_epochs}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    kaggle_path = r"C:\\Users\\anton\\Documents\\Datasets\\Kaggle Brain MRI"
    train_meta_on_kaggle_bmri(kaggle_path)

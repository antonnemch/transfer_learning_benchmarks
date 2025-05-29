# train_compare_meta_fft.py
# Compare FFT and MetaLR on 50% of Kaggle Brain MRI with batch size 16

import torch
from torch import nn
from models.resnet_base import resnet50_base
from utils.meta_train import get_optimizer, train_meta_step
from utils.loaders import load_kaggle_brain_mri_subset
from utils.train_resnet import train_one_epoch
from utils.train_resnet import count_parameters, count_parameters_by_module
import os

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

def initialize_model(num_classes, device):
    model = resnet50_base(pretrained=True, num_classes=num_classes)
    return model.to(device)

def train_fft(model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    print("\n=== Training with Full Fine-Tuning ===")
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"FFT Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"FFT Test Acc: {test_acc:.4f}")

def train_metalr(model, meta_model, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, hyper_lr):
    print("\n=== Training with MetaLR ===")
    now_lr = [0.1 * lr] * 18 + [lr]
    train_meta_step.meta_iterator = iter(val_loader)
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss, _, now_lr = train_meta_step(
                images, labels, model, meta_model, now_lr, val_loader,
                optimizer, criterion, epoch,
                batch_idx=None, num_batches=None
            )
        val_acc = evaluate_model(model, val_loader, device)
        print(f"MetaLR Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"MetaLR Test Acc: {test_acc:.4f}")

def run_comparison(kaggle_path, batch_size=16, num_epochs=5, lr=1e-2, hyper_lr=1e-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri_subset(
        kaggle_path, batch_size=batch_size, split_ratio=0.5)

    criterion = nn.CrossEntropyLoss()

    # FFT
    fft_model = initialize_model(num_classes, device)
    fft_optimizer = torch.optim.Adam(fft_model.parameters(), lr=lr)
    train_fft(fft_model, train_loader, val_loader, test_loader, criterion, fft_optimizer, device, num_epochs)

    # MetaLR
    meta_model = initialize_model(num_classes, device)
    metalr_model = initialize_model(num_classes, device)
    metalr_optimizer = get_optimizer(metalr_model, [0.1 * lr] * 18 + [lr])
    train_metalr(metalr_model, meta_model, train_loader, val_loader, test_loader,
                 criterion, metalr_optimizer, device, num_epochs, lr, hyper_lr)

if __name__ == "__main__":
    kaggle_path = r"C:\\Users\\anton\\Documents\\Datasets\\Kaggle Brain MRI"
    run_comparison(kaggle_path)

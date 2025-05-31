# train_kaggle_lora.py
# Train ResNet-50 with LoRA modules on Kaggle Brain MRI dataset

from models.lora_layers import initialize_lora_model
import torch
from torch import optim
from utils.dataset_loaders import load_kaggle_brain_mri
from utils.resnet_utils import train_one_epoch, count_parameters, count_parameters_by_module, evaluate_model
from utils.dataset_summarize import summarize
import os


def train_lora_on_kaggle_bmri(
    kaggle_path,
    batch_size=32,
    num_epochs=5,
    lr=1e-3,
    save_dir="."
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(kaggle_path, batch_size=batch_size)
    summarize("Kaggle Brain MRI", train_loader, val_loader, test_loader, num_classes)

    # Initialize model
    model = initialize_lora_model(num_classes, device)

    # Inspect parameters
    count_parameters(model)
    count_parameters_by_module(model)

    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_acc:.4f}")

    # Final evaluation
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Save model
    filename = f"resnet50_lora_bs{batch_size}_lr{lr}_ep{num_epochs}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    kaggle_path = r"C:\Users\anton\Documents\Datasets\Kaggle Brain MRI"
    train_lora_on_kaggle_bmri(kaggle_path)

# test_dataset_stats.py
# Loads all datasets using loaders.py and prints summary statistics

from utils.loaders import (
    load_kaggle_brain_mri,
    load_isic,
    load_pathmnist_npz
)
import os
import torch
from collections import Counter

# Provide paths to your datasets
kaggle_path = r"C:\Users\anton\Documents\Datasets\Kaggle Brain MRI"
isic_image_dir = r"C:\Users\anton\Documents\Datasets\ISIC\Images"
isic_label_csv = r"C:\Users\anton\Documents\Datasets\ISIC\ISIC2018_Task3_Training_GroundTruth.csv"
pathmnist_npz = r"C:\Users\anton\Documents\Datasets\PathMNIST\pathmnist_224.npz"

# Utility to count total samples in a DataLoader
def count_loader_samples(loader):
    return sum(len(batch[0]) for batch in loader)

# Utility to count class occurrences
def count_class_distribution(loader, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.int32)
    for images, labels in loader:
        for label in labels:
            class_counts[label] += 1
    return class_counts

# Wrapper to summarize dataset
def summarize(name, train_loader, val_loader, test_loader, num_classes):
    print(f"\n--- {name} Dataset Summary ---")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {count_loader_samples(train_loader)}")
    print(f"Val samples:   {count_loader_samples(val_loader)}")
    print(f"Test samples:  {count_loader_samples(test_loader)}")

    print("Class distribution (train):")
    print(count_class_distribution(train_loader, num_classes).tolist())

# Load and summarize all three datasets
train, val, test, n_cls = load_kaggle_brain_mri(kaggle_path)
summarize("Kaggle Brain MRI", train, val, test, n_cls)

train, val, test, n_cls = load_isic(isic_image_dir, isic_label_csv)
summarize("ISIC", train, val, test, n_cls)

train, val, test, n_cls = load_pathmnist_npz(pathmnist_npz)
summarize("PathMNIST", train, val, test, n_cls)

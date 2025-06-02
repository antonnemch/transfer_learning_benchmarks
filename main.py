import itertools
import torch
from torch import nn, optim
from train import train_models
from utils.dataset_loaders import load_kaggle_brain_mri

# === Load dataset ===
kaggle_path = r"C:\Users\anton\Documents\Datasets\Kaggle Brain MRI"

# === Models to run ===
run_models = {
    'fft': True,
    'metalr': True,
    'conv_adapters': True,
    'lora': True
}

# === Define hyperparameter search space ===
hyperparams = {
    'lr': [1e-3],
    'hyper_lr': [1e-1],
    'reduction': [64],
    'r': [16],
    'lora_alpha': [32],
    'num_epochs': [1],
    'batch_size': [32]
}

# === Fixed settings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# === Generate all combinations ===
keys, values = zip(*hyperparams.items())
param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
print("\n=== Grid Search Configurations ===")
for i, config in enumerate(param_grid):
    print(f"{i+1:>2}: {config}")

# === Run experiments ===
for params in param_grid:
    print("\n=== Running experiment with config: ===")
    print(params)

    train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(kaggle_path, params['batch_size'],subset_fraction=0.2)
    optimizer_class = optim.Adam
    

    train_models(
        run_fft=run_models['fft'],
        run_metalr=run_models['metalr'],
        run_conv_adapters=run_models['conv_adapters'],
        run_lora=run_models['lora'],
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer_class,
        device=device,
        num_epochs=params['num_epochs'],
        lr=params['lr'],
        hyper_lr=params['hyper_lr'],
        reduction=params['reduction'],
        r=params['r'],
        lora_alpha=params['lora_alpha']
    )

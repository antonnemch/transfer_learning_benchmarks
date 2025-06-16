import itertools
import time
import torch
from torch import nn, optim
from train import train_models
from utils.dataset_loaders import load_kaggle_brain_mri
from utils.dataset_summarize import summarize_log

# === Load dataset ===
kaggle_path = r"datasets\Kaggle Brain MRI"

# === Models to run ===
run_models = {
    'fft': True,
    'metalr': True,
    'conv_adapters': True,
    'lora': True,
    'spline': False
}

# === Define hyperparameter search space ===
hyperparams = {
    'lr': [1e-3, 1e-1],
    'num_epochs': [30],
    'batch_size': [64, 32],
    'data_subset': [0.05, 0.2, 0.5, 1],
    'hyper_lr': [1e-1, 1e-3],
    'reduction': [64, 32],
    'r': [64, 32, 16, 8],
    'lora_alpha': [64, 32, 16]
}

# === Define hyperparameter search space for testing ===
hyperparams = {
    'lr': [1e-1],
    'num_epochs': [1],
    'batch_size': [64],
    'data_subset': [0.05],
    'hyper_lr': [1e-1],
    'reduction': [64], 
    'r': [64],
    'lora_alpha': [64],
    'sharing': ['model','layer','block','channel'] # For splines, can be 'channel', 'layer', 'block', 'model' ['layer','channel','block','model']
}

# === Fixed settings ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"\n================ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"\n================ Using CPU")
    
criterion = nn.CrossEntropyLoss()

# === Param relevance mapping per model ===
model_param_map = {
    "fft": {"lr", "num_epochs", "batch_size", "data_subset"},
    "metalr": {"lr", "hyper_lr", "num_epochs", "batch_size", "data_subset"},
    "conv_adapters": {"lr", "reduction", "num_epochs", "batch_size", "data_subset"},
    "lora": {"lr", "r", "lora_alpha", "num_epochs", "batch_size", "data_subset"},
    "spline": {"lr", "num_epochs", "batch_size", "data_subset","sharing"}
}

# === Precompute total number of configurations across all models ===
total_experiments = 0
exp_i = 0
for model_name, should_run in run_models.items():
    if not should_run:
        continue
    relevant_params = model_param_map[model_name]
    param_counts = [len(hyperparams[p]) for p in relevant_params]
    model_total = 1
    for count in param_counts:
        model_total *= count
    total_experiments += model_total

print(f"\n=== TOTAL EXPERIMENTS TO RUN: {total_experiments} ===\n")

# === Ensure reproducible dataset splits across all models and configs ===
torch.manual_seed(42)
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
print(f"=== Timestamp for this run: {timestamp} ===\n")

# === Run model-specific grid searches ===
for model_name, should_run in run_models.items():
    if not should_run:
        continue
       
    relevant_params = sorted(model_param_map[model_name]) 
    param_values = [hyperparams[p] for p in relevant_params]
    param_names = relevant_params
    model_param_combinations = [dict(zip(param_names, combination)) for combination in itertools.product(*param_values)]

    total_configs = len(model_param_combinations)

    for i, config in enumerate(model_param_combinations):
        config['seed'] = 42
        exp_i += 1 
        print(f"\n[{model_name.upper()}] (Exp. {exp_i}/{total_experiments}) Config {i+1}/{total_configs}: {config}")

        train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(
            kaggle_path,
            batch_size=config['batch_size'],
            subset_fraction=config['data_subset']
        )
        dataset_summary = summarize_log("Kaggle Brain MRI", train_loader, val_loader, test_loader, num_classes)

        optimizer_class = optim.Adam

        train_models(
            run_fft=(model_name == "fft"),
            run_metalr=(model_name == "metalr"),
            run_conv_adapters=(model_name == "conv_adapters"),
            run_lora=(model_name == "lora"),
            run_spline=(model_name == "spline"),
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer_class,
            device=device,
            num_epochs=config['num_epochs'],
            lr=config['lr'],
            hyper_lr=config.get('hyper_lr', 0.1),
            reduction=config.get('reduction', 64),
            r=config.get('r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            config=config,
            dataset_summary=dataset_summary,
            timestamp=timestamp,
            sharing=config.get('sharing', 'channel')
        )

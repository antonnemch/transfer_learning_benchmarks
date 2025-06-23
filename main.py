import itertools
import time
import torch
from torch import nn, optim
from train import safe_train
from utils.dataset_loaders import load_kaggle_brain_mri
from utils.dataset_summarize import summarize_log
from utils.resnet_utils import compute_num_experiments
# === Load dataset ===
kaggle_path = r"datasets\Kaggle Brain MRI"


# === Define hyperparameter search space ===
hyperparams = {
    'lr': [1e-4],
    'num_epochs': [1],
    'batch_size': [64],
    'data_subset': [0.1],
    'activation_type': ["laplacian_gpaf", "all_3x3_shared","laplacian_gpaf_by_model"] #"full_relu",
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
    "GPAF": {"lr", "num_epochs", "batch_size", "data_subset","activation_type"},
}

total_experiments = compute_num_experiments(model_name="GPAF", hyperparams=hyperparams, model_param_map=model_param_map)
print(f"\n=== TOTAL EXPERIMENTS TO RUN: {total_experiments} ===\n")

# === Ensure reproducible dataset splits across all models and configs ===
torch.manual_seed(42)
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
print(f"=== Timestamp for this run: {timestamp} ===\n")
exp_i = 0


model_name = "GPAF"    
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
    safe_train("GPAF", timestamp, config ,dataset_summary,
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer = optimizer_class,
        device=device,
        num_epochs = config['num_epochs'],
        lr = config['lr'],
        activation_type = config.get('activation_type', None))
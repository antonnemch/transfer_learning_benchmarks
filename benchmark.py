import os
import time
import random
import numpy as np
import torch

from train import compute_total_experiments, train_conv_adapter, train_gpaf, train_lora, get_model_param_combinations
from utils.dataset_loaders import load_kaggle_brain_mri
from utils.dataset_summarize import summarize_log
from utils.ExcelTrainingLogger import make_logger

# === Unified Hyperparameter Grid ===
hyperparams = {
    'model': ['GPAF', 'ConvAdapter', 'LoRA'],
    'net_lr': [1e-3],
    'num_epochs': [30],
    'batch_size': [16],
    'data_subset': [0.01, 0.05, 0.1, 0.5],  # Subset fractions for training
    'activation_type': ['stage4.2_act2only_channelwise_kglap','full_relu',
                        'stage4.2_act123_channelwise_kglap', 'stage3_4_act2_blockshared_kglap',
                        'stage4_act2only_channelwise_kglap'],
    'act_lr': [1e-5, None],  # None for no activation learning rate
    'act_optimizer': ['adam'],
    'reduction': [64],  # For ConvAdapter
    'r': [64],          # For LoRA
    'lora_alpha': [32], # For LoRA
    'modifiers': [
        {"TrainBN": False, "Deferred": None, "Regularization": None},  # Default settings
        {"TrainBN": False, "Deferred": None, "Regularization": 1e-2}, # Extra regularization
        {"TrainBN": False, "Deferred": 3, "Regularization": None}, # Just deferred activation training
        {"TrainBN": False, "Deferred": None, "Regularization": 1e-4}, # Just Regularization
        {"TrainBN": True, "Deferred": None, "Regularization": None},  # Train BatchNorm
        {"TrainBN": False, "Deferred": 3, "Regularization": 1e-4}, # Deferred + Regularization
        {"TrainBN": True, "Deferred": 3, "Regularization": None},   # Train BatchNorm + Deferred
        {"TrainBN": True, "Deferred": None, "Regularization": 1e-4}, # Train BatchNorm + Regularization
        {"TrainBN": True, "Deferred": 3, "Regularization": 1e-4},   # Train BatchNorm + Deferred + Regularization
    ]
}

# === Model Param Map ===
model_param_map = {
    "GPAF": {"net_lr", "num_epochs", "batch_size", "data_subset", "activation_type", "act_lr", "act_optimizer", "modifiers"},
    "ConvAdapter": {"net_lr", "num_epochs", "batch_size", "data_subset", "reduction"},
    "LoRA": {"net_lr", "num_epochs", "batch_size", "data_subset", "r", "lora_alpha"},
}

# === Device and Seed Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# === Unified Logger (single Excel file for all runs) ===
timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
excel_log_path = os.path.join("logs", f"UNIFIED_RESULTS_{timestamp}.xlsx")



os.makedirs("logs", exist_ok=True)
total_experiments = compute_total_experiments(hyperparams, model_param_map)
exp_i = 0
for model_name in ['GPAF', 'ConvAdapter', 'LoRA']:
    configs = get_model_param_combinations(model_name, model_param_map, hyperparams)
    for config in configs:
        exp_i += 1
        # Add model name to config dict for logging
        config_with_model = dict(config)
        config_with_model['model'] = model_name
        print(f"\n[{model_name}] (Exp. {exp_i}/{total_experiments}) Config: {config_with_model}")
        train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(
            os.path.join('datasets', 'Kaggle Brain MRI'),
            batch_size=config['batch_size'],
            subset_fraction=config['data_subset']
        )
        dataset_summary = summarize_log("Kaggle Brain MRI", train_loader, val_loader, test_loader, num_classes)
        logger = make_logger("UNIFIED", config=config_with_model, timestamp=timestamp, base_dir="logs")
        logger.log_dataset_summary(dataset_summary)
        logger.log_hyperparams()
        if model_name == "GPAF":
            train_gpaf(config_with_model, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger,device)
        elif model_name == "ConvAdapter":
            train_conv_adapter(config_with_model, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device)
        elif model_name == "LoRA":
            train_lora(config_with_model, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device)
        logger.save()

print(f"\nAll experiments complete. Results saved to {excel_log_path}")

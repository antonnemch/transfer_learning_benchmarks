import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import itertools
import time
import random, numpy as np
import torch
from torch import nn, optim
from train import safe_train
from utils.dataset_loaders import load_kaggle_brain_mri
from utils.dataset_summarize import summarize_log
from utils.resnet_utils import compute_num_experiments
import os  # Ensure os is imported at the top
# === Load dataset ===
kaggle_path = os.path.join('datasets', 'Kaggle Brain MRI')


# === Define hyperparameter search space ===
hyperparams_1 = {
    'net_lr': [1e-3,1e-4],
    'act_lr': [1e-5, 1e-6, None], # None
    'num_epochs': [30],
    'batch_size': [64],
    'data_subset': [0.05,0.1,0.5], 
    'act_optimizer': ["adam","adadelta"],  # NEW: add optimizer choice ["adam", "adadelta"]
    'activation_type': 
        ["full_relu",
        # KGActivationLaplacian (kglap)
        "all_act123_shared_kglap",
        "stage_act123_shared_kglap",
        "stagegroup_act2only_shared_kglap",
        "stage3_4_act2_blockshared_kglap",
        "stage4_act2only_channelwise_kglap",
        "all_act123_channelwise_kglap",

        # KGActivationGeneral (kggen)
        "all_act123_shared_kggen",
        "stage_act123_shared_kggen",
        "act13_shared_kggen",
        "block_act2_shared_kggen",
        "stage3_4_act2_blockshared_kggen",
        "all_act123_channelwise_kggen",

        # Baselines
        "all_act123_channelwise_prelu",
        "all_act123_shared_swishfixed",
        "all_act123_channelwise_swishlearn"],
}

hyperparams_2 = {
    'net_lr': [1e-3],
    'act_lr': [1e-5], # None
    'num_epochs': [30],
    'batch_size': [32],
    'data_subset': [0.05,0.1,0.2,0.5], 
    'act_optimizer': ["adam","adadelta"],  # NEW: add optimizer choice ["adam", "adadelta"]
    'activation_type': 
        ["full_relu",
        # KGActivationLaplacian (kglap)
        "stage3_4_act2_blockshared_kglap",
        "stage4_act2only_channelwise_kglap",
        "stagegroup_act2only_shared_kglap",

        # Baselines
        "act2only_channelwise_prelu",
        "act2only_shared_swishfixed",
        "all_act123_channelwise_swishlearn",
        "stage3_4_act2_blockshared_swishlearn"],
}

# Testing theory that a fixed low learning rate for activations is better, 
# AND that activations should not be modified too far away from the fully connected layer
hyperparams = {
    'net_lr': [1e-3],
    'act_lr': [1e-6], # Decided to use an even lower learning rate for activations
    'num_epochs': [30],
    'batch_size': [32],
    'data_subset': [0.02,0.05,0.1,0.2,0.5, 1.0],  # Added 1.0 for full dataset
    'act_optimizer': ["adam"],  # removed "adadelta" temporarily to limit experiment count
    'activation_type': 
        ["full_relu",
        # KGActivationLaplacian (kglap)
        # compare results to hyperparams_2 to understand act_lr impact
        "stage3_4_act2_blockshared_kglap", 
        "stage4_act2only_channelwise_kglap",
        "stagegroup_act2only_shared_kglap", 
        # NEW KGLAP Testing for only late stage activation modification
        "stage4.2_act2only_channelwise_kglap",
        "stage4.2_act123_channelwise_kglap",

        # Baselines
        # "act2only_channelwise_prelu",
        "act2only_shared_swishfixed", # only kept best performing baseline
        #"stage3_4_act2_blockshared_swishlearn"
        ],
}

# Testing regularization, bias enablement, deferred activation learning, and adadelta optimizer
hyperparams_3 = {
    'net_lr': [1e-3],
    'act_lr': [1e-6], # Decided to use an even lower learning rate for activations
    'num_epochs': [30],
    'batch_size': [32],
    'data_subset': [0.02,0.1,0.2,0.5],  # Limited data subset for faster experiments
    'act_optimizer': ["adam","adadelta"],  # Testing both Adam and Adadelta optimizers
    'activation_type': 
        ["full_relu",
        # KGActivationLaplacian (kglap)
        # compare results to hyperparams_2 to understand act_lr impact
        "stage3_4_act2_blockshared_kglap", 
        "stage4_act2only_channelwise_kglap",
        "stagegroup_act2only_shared_kglap", 
        # NEW KGLAP Testing for only late stage activation modification
        "stage4.2_act2only_channelwise_kglap",
        "stage4.2_act123_channelwise_kglap",

        # Baselines
        # "act2only_channelwise_prelu",
        "act2only_shared_swishfixed", # only kept best performing baseline
        #"stage3_4_act2_blockshared_swishlearn"
        ],
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
    "GPAF": {"net_lr", "num_epochs", "batch_size", "data_subset","activation_type","act_lr", "act_optimizer"},
}

total_experiments = compute_num_experiments(model_name="GPAF", hyperparams=hyperparams, model_param_map=model_param_map)
print(f"\n=== TOTAL EXPERIMENTS TO RUN: {total_experiments} ===\n")

# === Ensure reproducible dataset splits across all models and configs ===
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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

    # Pass the actual optimizer class, not a lambda, so train_GPAF can use it for act_optimizer
    if config['act_optimizer'].lower() == 'adam':
        act_optimizer_class = optim.Adam
    elif config['act_optimizer'].lower() == 'adadelta':
        act_optimizer_class = optim.Adadelta
    else:
        raise ValueError(f"Unknown act_optimizer: {config['act_optimizer']}")

    safe_train("GPAF", timestamp, config ,dataset_summary,
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer = act_optimizer_class,
        device=device,
        num_epochs = config['num_epochs'],
        net_lr = config['net_lr'],
        act_lr = config['act_lr'],
        activation_type = config.get('activation_type', None))
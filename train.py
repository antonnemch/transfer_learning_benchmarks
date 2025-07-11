import itertools
import os
import time
from torch import torch, nn, optim

from models.resnet_base import initialize_basic_model
from utils.resnet_utils import EarlyStopping, build_activation_map, evaluate_model, train_one_epoch
from models.resnet_base import initialize_basic_model
from models.conv_adapter_module import initialize_conv_model
from models.lora_layers import initialize_lora_model
from models.activation_configs import activations


# === Experiment Grid Generation ===
def get_model_param_combinations(model_name,model_param_map, hyperparams):
    relevant_params = sorted(model_param_map[model_name])
    param_values = [hyperparams[p] for p in relevant_params]
    param_names = relevant_params
    return [dict(zip(param_names, combination)) for combination in itertools.product(*param_values)]

# === Model Training Dispatch ===
def train_gpaf(config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device):
    os.makedirs(os.path.join("saved_models"), exist_ok=True)
    model = initialize_basic_model(num_classes, device, freeze=True)
    net_optimizer = optim.Adam(model.parameters(), lr=config['net_lr'])
    act_optimizer = None
    if config.get('act_lr'):
        act_optimizer = getattr(optim, config['act_optimizer'].capitalize())(model.parameters(), lr=config['act_lr'])
    activation_map = build_activation_map(activations[config['activation_type']])
    modifiers = config.get('modifiers', {})
    deferred_epochs = modifiers.get('Deferred', None)
    train_bn = modifiers.get('TrainBN', False)
    # Always initialize early_stopper before training loop
    early_stopper = EarlyStopping(patience=5)
    if deferred_epochs is None or deferred_epochs == 0:
        model.set_custom_activation_map(activation_map, train_bn=train_bn)
        activation_map_set = True
        logger.log_param_counts(model)
    else:
        activation_map_set = False
        print(f"Deferring activation map set for {deferred_epochs} epochs.")
    print("\n=== Training with GPAF ===")
    best_val_acc = -float("inf")
    best_model_state = None
    best_epoch = -1
    early_stop_pending = False
    for epoch in range(config['num_epochs']):
        try:
            # Set activation map if deferred and epoch reached
            if (
                not activation_map_set
                and deferred_epochs is not None
                and epoch >= deferred_epochs
            ):
                model.set_custom_activation_map(activation_map, train_bn=train_bn)
                print(
                    f"Custom activation map set at epoch {epoch+1} (deferred {deferred_epochs} epochs)"
                )
                early_stopper = EarlyStopping(patience=5)
                logger.log_param_counts(model)
                activation_map_set = True
                # If early stopping was pending, allow at least one more epoch after activation map is set
                if early_stop_pending:
                    early_stop_pending = False
            start = time.time()
            # Training step
            train_loss, acc = train_one_epoch(
                model,
                train_loader,
                net_optimizer,
                device,
                nn.CrossEntropyLoss(),
                epoch,
                logger,
                act_optimizer,
                modifiers,
            )
            # Validation step
            val_loss, val_acc = evaluate_model(
                model, val_loader, device, nn.CrossEntropyLoss(), logger, epoch
            )
            # Logging
            elapsed = time.time() - start
            logger.log_epoch_metrics(
                epoch, train_loss, val_loss, acc, elapsed, torch.cuda.max_memory_allocated()
            )
            print(
                f"GPAF Epoch {epoch+1}/{config['num_epochs']} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}"
            )
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_epoch = epoch
            # Early stopping check
            if early_stopper.step(val_loss):
                if not activation_map_set:
                    # Defer early stopping until after activation map is set
                    print(f"Early stopping triggered at epoch {epoch+1}, but activation map not set yet. Continuing until after activation map is set.")
                    early_stop_pending = True
                else:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        except Exception as e:
            print(f"[ERROR][GPAF][Epoch {epoch+1}] {e}")
            import traceback
            traceback.print_exc()
            break
    # Save best model weights
    best_model_path = os.path.join(
        "saved_models", f"{logger.model_name}_{logger.config_id}_best.pt"
    )
    torch.save(best_model_state, best_model_path)
    print(f"Best model (epoch {best_epoch+1}) saved to {best_model_path}")
    # Load best model for test evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    # Final test evaluation and logging (handled by evaluate_model)
    try:
        test_acc = evaluate_model(
            model, test_loader, device, nn.CrossEntropyLoss(), logger, phase='test'
        )
        print(f"GPAF Test Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"[ERROR][GPAF][Test] {e}")
        import traceback
        traceback.print_exc()
        test_acc = None
    # Save final model (last epoch)
    final_model_path = os.path.join(
        "saved_models", f"{logger.model_name}_{logger.config_id}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.log_model_path(final_model_path)
    # Optionally log final result for unified logger compatibility
    if hasattr(logger, 'log_final_result'):
        logger.log_final_result(model_name="GPAF", config=config, test_acc=test_acc)


def train_conv_adapter(config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device):
    os.makedirs(os.path.join("saved_models"), exist_ok=True)
    model = initialize_conv_model(num_classes, device, reduction=config['reduction'])
    optimizer = optim.Adam(model.parameters(), lr=config['net_lr'])
    early_stopper = EarlyStopping(patience=5)
    logger.log_param_counts(model)
    print("\n=== Training with ConvAdapter ===")
    best_val_acc = -float("inf")
    best_model_state = None
    best_epoch = -1
    for epoch in range(config['num_epochs']):
        try:
            start = time.time()
            train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, nn.CrossEntropyLoss(), epoch, logger)
            val_loss, val_acc = evaluate_model(model, val_loader, device, nn.CrossEntropyLoss(), logger, epoch)
            elapsed = time.time() - start
            logger.log_epoch_metrics(epoch, train_loss, val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
            print(f"ConvAdapter Epoch {epoch+1}/{config['num_epochs']} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_epoch = epoch
            if early_stopper.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        except Exception as e:
            print(f"[ERROR][ConvAdapter][Epoch {epoch+1}] {e}")
            import traceback
            traceback.print_exc()
            break
    best_model_path = os.path.join(
        "saved_models", f"{logger.model_name}_{logger.config_id}_best.pt"
    )
    torch.save(best_model_state, best_model_path)
    print(f"Best model (epoch {best_epoch+1}) saved to {best_model_path}")
    # Final test evaluation and logging (handled by evaluate_model)
    try:
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_acc = evaluate_model(
            model, test_loader, device, nn.CrossEntropyLoss(), logger, phase='test'
        )
        print(f"ConvAdapter Test Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"[ERROR][ConvAdapter][Test] {e}")
        import traceback
        traceback.print_exc()
        test_acc = None
    final_model_path = os.path.join(
        "saved_models", f"{logger.model_name}_{logger.config_id}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.log_model_path(final_model_path)
    if hasattr(logger, 'log_final_result'):
        logger.log_final_result(model_name="ConvAdapter", config=config, test_acc=test_acc)


def train_lora(config, train_loader, val_loader, test_loader, num_classes, dataset_summary, logger, device):
    os.makedirs(os.path.join("saved_models"), exist_ok=True)
    lora_config = {"r": config['r'], "lora_alpha": config['lora_alpha'], "lora_dropout": 0, "merge_weights": True}
    model = initialize_lora_model(num_classes, device, lora_config=lora_config)
    optimizer = optim.Adam(model.parameters(), lr=config['net_lr'])
    early_stopper = EarlyStopping(patience=3)
    logger.log_param_counts(model)
    print("\n=== Training with LoRA ===")
    best_val_acc = -float("inf")
    best_model_state = None
    best_epoch = -1
    for epoch in range(config['num_epochs']):
        try:
            start = time.time()
            train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, nn.CrossEntropyLoss(), epoch, logger)
            val_loss, val_acc = evaluate_model(model, val_loader, device, nn.CrossEntropyLoss(), logger, epoch)
            elapsed = time.time() - start
            logger.log_epoch_metrics(epoch, train_loss, val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
            print(f"LoRA Epoch {epoch+1}/{config['num_epochs']} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict()
                best_epoch = epoch
            if early_stopper.step(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        except Exception as e:
            print(f"[ERROR][LoRA][Epoch {epoch+1}] {e}")
            import traceback
            traceback.print_exc()
            break
    best_model_path = os.path.join(
        "saved_models", f"{logger.model_name}_{logger.config_id}_best.pt"
    )
    torch.save(best_model_state, best_model_path)
    print(f"Best model (epoch {best_epoch+1}) saved to {best_model_path}")
    # Final test evaluation and logging (handled by evaluate_model)
    try:
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_acc = evaluate_model(
            model, test_loader, device, nn.CrossEntropyLoss(), logger, phase='test'
        )
        print(f"LoRA Test Acc: {test_acc:.4f}")
    except Exception as e:
        print(f"[ERROR][LoRA][Test] {e}")
        import traceback
        traceback.print_exc()
        test_acc = None
    final_model_path = os.path.join(
        "saved_models", f"{logger.model_name}_{logger.config_id}_final.pt"
    )
    torch.save(model.state_dict(), final_model_path)
    logger.log_model_path(final_model_path)
    if hasattr(logger, 'log_final_result'):
        logger.log_final_result(model_name="LoRA", config=config, test_acc=test_acc)


# === Compute total number of experiments ===
def compute_total_experiments(hyperparams, model_param_map, run_models=None):
    total = 0
    for model_name in ['GPAF', 'ConvAdapter', 'LoRA']:
        if run_models is not None and not run_models.get(model_name, True):
            continue
        relevant_params = model_param_map[model_name]
        param_counts = [len(hyperparams[p]) for p in relevant_params]
        model_total = 1
        for count in param_counts:
            model_total *= count
        total += model_total
    return total
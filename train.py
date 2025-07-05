import time
from xml.parsers.expat import model
import torch
import os
import traceback
from models.resnet_base import initialize_basic_model
from utils.ExcelTrainingLogger import make_logger
from utils.resnet_utils import EarlyStopping, count_parameters, count_parameters_by_module, print_model_activations, train_one_epoch
from utils.resnet_utils import evaluate_model, build_activation_map, print_activation_map
from models.activation_configs import activations
from torchsummary import summary


def train_GPAF(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, logger, activation_type, net_lr, act_lr = None):
    modifiers = {
        "biases": False,  # Do not modify biases
        "Deferred": False,  # Use deferred activation updates
        "Regularization": False,  # Use regularization
    }

    os.makedirs(os.path.join("saved_models"), exist_ok=True)
    model = initialize_basic_model(num_classes, device, freeze=True)

    # Always use Adam for network optimizer
    network_optimizer = torch.optim.Adam(model.parameters(), lr=net_lr)

    # Use the passed optimizer type for activation optimizer if act_lr is provided
    if act_lr:
        activation_optimizer = optimizer(model.parameters(), lr=act_lr)
    else:
        activation_optimizer = None

    early_stopper = EarlyStopping(patience=3)  # Early stopping instance

    activation_map = build_activation_map(activations[activation_type])

    print_activation_map(activation_map)
    model.set_custom_activation_map(activation_map)
    logger.log_param_counts(model)


    print("\n=== Training with GPAF ===")
    best_val_acc = -float('inf')
    best_model_state = None
    best_epoch = -1
    for epoch in range(num_epochs):
        start = time.time()
        # Training step
        train_loss, acc = train_one_epoch(model, train_loader, network_optimizer, device, criterion, epoch, logger, activation_optimizer)
        # Validation step
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, logger, epoch)
        # Logging
        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, train_loss,val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"GPAF Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            best_epoch = epoch
        # Early stopping check
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Save best model weights
    best_model_path = os.path.join("saved_models", f"{logger.model_name}_{logger.config_id}_best.pt")
    torch.save(best_model_state, best_model_path)
    print(f"Best model (epoch {best_epoch+1}) saved to {best_model_path}")
    # Load best model for test evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    # Final test evaluation
    test_acc = evaluate_model(model, test_loader, device, criterion, logger ,phase="test")
    print(f"GPAF Test Acc: {test_acc:.4f}")
    # Save final model (last epoch)
    final_model_path = os.path.join("saved_models", f"{logger.model_name}_{logger.config_id}_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.log_model_path(final_model_path)


def safe_train(model_name, timestamp, config, dataset_summary, **kwargs):
    try:
        logger = make_logger(model_name, config=config, timestamp=timestamp)  
        logger.log_dataset_summary(dataset_summary)
        logger.log_hyperparams()
        train_GPAF(**kwargs, logger=logger)
    except Exception as e:
        print(f"[ERROR] {model_name} failed: {e}")
        traceback.print_exc()
    finally:
        # Ensure logger is saved even if training fails
        if 'logger' in locals():
            logger.save()
        else:
            print("Logger not initialized, skipping save.")
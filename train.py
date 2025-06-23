import time
import torch
import traceback
from models.resnet_base import initialize_basic_model
from utils.ExcelTrainingLogger import make_logger
from utils.resnet_utils import EarlyStopping, count_parameters, count_parameters_by_module, print_model_activations, train_one_epoch
from utils.resnet_utils import evaluate_model, build_activation_map, print_activation_map
from models.custom_activations import activations

def train_GPAF(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, logger, activation_type):
    model = initialize_basic_model(num_classes, device,freeze=True)
    optimizer = optimizer(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=3)  # Early stopping instance
    logger.log_param_counts(model)

    activation_map = build_activation_map(activations[activation_type])

    print_activation_map(activation_map)

    model.set_custom_activation_map(activation_map)


    if True:
        count_parameters(model)
        count_parameters_by_module(model)
        #print(f"\n=== Model Summary ===\n{model}\n")
        print_model_activations(model)


    print("\n=== Training with GPAF ===")
    for epoch in range(num_epochs):
        start = time.time()
        # Training step
        train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, logger)
        # Validation step
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, logger, epoch)
        # Logging
        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, train_loss,val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"GPAF Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
        # Early stopping check
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Final test evaluation
    test_acc = evaluate_model(model, test_loader, device, criterion, logger ,phase="test")
    print(f"GPAF Test Acc: {test_acc:.4f}")
    # Save model
    model_path = f"saved_models/{logger.model_name}_{logger.config_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()


def safe_train(model_name, timestamp, config, dataset_summary, **kwargs):
    try:
        logger = make_logger(model_name, config=config, timestamp=timestamp)  
        logger.log_dataset_summary(dataset_summary)
        logger.log_hyperparams()
        train_GPAF(**kwargs, logger=logger)
    except Exception as e:
        print(f"[ERROR] {model_name} failed: {e}")
        traceback.print_exc()
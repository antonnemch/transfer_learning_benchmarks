import time
import torch
import traceback
from models.conv_adapter_module import initialize_conv_model
from models.lora_layers import initialize_lora_model
from models.resnet_base import initialize_basic_model
from models.resnet_deepspline import initialize_spline_model
from utils.ExcelTrainingLogger import make_logger
from utils.meta_utils import train_meta_step
from utils.resnet_utils import EarlyStopping, count_parameters, count_parameters_by_module, train_one_epoch, train_one_epoch_spline
from utils.resnet_utils import evaluate_model

def train_GPAF(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, logger,gpaf):
    model = initialize_basic_model(num_classes, device,gpaf = gpaf,freeze=True)
    optimizer = optimizer(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=30)  # Early stopping instance
    logger.log_param_counts(model)
    if True:
        count_parameters(model)
        count_parameters_by_module(model)

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
    test_acc = evaluate_model(model, test_loader, device,phase="test")
    print(f"GPAF Test Acc: {test_acc:.4f}")
    # Save model
    model_path = f"saved_models/{logger.model_name}_{logger.config_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

def train_fft(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, logger):
    model = initialize_basic_model(num_classes, device)
    optimizer = optimizer(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=3)  # Early stopping instance
    logger.log_param_counts(model)

    print("\n=== Training with Full Fine-Tuning ===")
    for epoch in range(num_epochs):
        start = time.time()
        # Training step
        train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, logger)
        # Validation step
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, logger, epoch)
        # Logging
        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, train_loss,val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"FFT Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
        # Early stopping check
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Final test evaluation
    test_acc = evaluate_model(model, test_loader, device,phase="test")
    print(f"FFT Test Acc: {test_acc:.4f}")
    # Save model
    model_path = f"saved_models/{logger.model_name}_{logger.config_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

def train_metalr(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, hyper_lr, logger):
    model = initialize_basic_model(num_classes, device)
    meta_model = initialize_basic_model(num_classes, device)
    optimizer = optimizer(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=3)  # Early stopping instance
    logger.log_param_counts(model)

    print("\n=== Training with MetaLR ===")
    now_lr = [0.1 * lr] * 18 + [lr]
    train_meta_step.meta_iterator = iter(val_loader)

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        num_batches = len(train_loader)
        epoch_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            loss, _, now_lr = train_meta_step(
                images, labels, model, meta_model, now_lr, val_loader,
                optimizer, criterion, epoch,
                logger=logger, hyper_lr=hyper_lr,
                batch_idx=batch_idx, num_batches=num_batches
            )
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / num_batches
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, logger, epoch)

        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, avg_train_loss, val_loss, val_acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"MetaLR Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")

        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Final test evaluation
    test_acc = evaluate_model(model, test_loader, device, phase="test")
    print(f"MetaLR Test Acc: {test_acc:.4f}")

    model_path = f"saved_models/{logger.model_name}_{logger.config_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

def train_conv_adapters(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, reduction, lr, logger):
    model = initialize_conv_model(num_classes, device, reduction=reduction)
    optimizer = optimizer(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=3)  # Early stopping instance
    logger.log_param_counts(model)
    print("\n=== Training with Conv-Adapters ===")
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, logger)
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, logger, epoch)
        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, train_loss,val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"Conv Epoch {epoch}/{num_epochs} - Val Acc: {val_acc:.4f}")
        # Early stopping check
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    test_acc = evaluate_model(model, test_loader, device,phase="test")
    print(f"Conv Test Acc: {test_acc:.4f}")
    model_path = f"saved_models/{logger.model_name}_{logger.config_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

def train_lora(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, r, lora_alpha, lr, logger):
    lora_config = {
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0,
        "merge_weights": True
    }
    model = initialize_lora_model(num_classes, device, lora_config=lora_config)
    optimizer = optimizer(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=3)  # Early stopping instance
    logger.log_param_counts(model)
    print("\n=== Training with Lora-C ===")
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, acc = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, logger)
        val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, logger, epoch)
        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, train_loss,val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"Lora-C Epoch {epoch}/{num_epochs} - Val Acc: {val_acc:.4f}")
        # Early stopping check
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    test_acc = evaluate_model(model, test_loader, device,phase="test")
    print(f"Lora-C Test Acc: {test_acc:.4f}")
    model_path = f"saved_models/{logger.model_name}_{logger.config_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

def train_spline(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, logger, sharing="channel"):
    model = initialize_spline_model(num_classes, device, freeze=True, sharing=sharing)
    main_optimizer = optimizer(model.parameters_no_deepspline(), lr=lr)
    aux_optimizer = torch.optim.Adam(model.parameters_deepspline())
    early_stopper = EarlyStopping(patience=3)
    logger.log_param_counts(model)
    if True:
        count_parameters(model)
        count_parameters_by_module(model)

    print("\n=== Training with B-Spline Activation ===")
    index_error_count = 0
    max_index_errors = 10

    for epoch in range(num_epochs):
        start = time.time()

        try:
            train_loss, acc = train_one_epoch_spline(model, train_loader, [main_optimizer, aux_optimizer], device, criterion, epoch, logger)
        except IndexError as e:
            index_error_count += 1
            print(f"[WARNING] IndexError during training (epoch {epoch+1}): {e}")
            if index_error_count >= max_index_errors:
                print("[ABORTING] Too many IndexErrors during training.")
                return
            continue  # Skip to next epoch

        try:
            val_loss, val_acc = evaluate_model(model, val_loader, device, criterion, logger, epoch)
        except IndexError as e:
            index_error_count += 1
            print(f"[WARNING] IndexError during validation (epoch {epoch+1}): {e}")
            if index_error_count >= max_index_errors:
                print("[ABORTING] Too many IndexErrors during validation.")
                return
            continue  # Skip to next epoch

        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, train_loss, val_loss, acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"Spline Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f} - Val Loss: {val_loss:.4f}")
        if early_stopper.step(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    try:
        test_acc = evaluate_model(model, test_loader, device, phase="test")
        print(f"Spline Test Acc: {test_acc:.4f}")
    except IndexError as e:
        print(f"[ERROR] IndexError during final test evaluation: {e}")
        test_acc = None

    model_path = f"saved_models/{logger.model_name}_{logger.config_id}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()


def train_models(run_fft, run_metalr, run_conv_adapters, run_lora, run_spline, run_gpaf,
                 num_classes, train_loader, val_loader, test_loader, 
                 criterion, optimizer, device, num_epochs, lr, hyper_lr, 
                 reduction, r, lora_alpha, activation, timestamp = "", config=None, dataset_summary=None,sharing="channel"):
    
    def safe_train(model_name, timestamp, train_fn, **kwargs):
        try:
            logger = make_logger(model_name, config=config, timestamp=timestamp)  
            logger.log_dataset_summary(dataset_summary)
            logger.log_hyperparams()
            train_fn(**kwargs, logger=logger)
        except Exception as e:
            print(f"[ERROR] {model_name} failed: {e}")
            traceback.print_exc()
    if run_gpaf:
        safe_train("GPAF", timestamp, train_GPAF,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            lr=lr,gpaf=activation)
        
    if run_fft:
        safe_train("FFT", timestamp, train_fft,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            lr=lr)

    if run_metalr:
        safe_train("MetaLR", timestamp, train_metalr,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            hyper_lr=hyper_lr)

    if run_conv_adapters:
        safe_train("ConvAdapter", timestamp, train_conv_adapters,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            reduction=reduction,
            lr=lr)

    if run_lora:
        safe_train("LoRA-C", timestamp, train_lora,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            r=r,
            lora_alpha=lora_alpha,
            lr=lr)

    if run_spline:
        safe_train("DeepSpline", timestamp, train_spline,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            lr=lr,sharing=sharing)
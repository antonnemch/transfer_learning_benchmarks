import time
import torch
from models.conv_adapter_module import initialize_conv_model
from models.lora_layers import initialize_lora_model
from models.resnet_base import initialize_basic_model
from utils.ExcelTrainingLogger import make_logger
from utils.meta_utils import train_meta_step
from utils.resnet_utils import train_one_epoch
from utils.resnet_utils import evaluate_model


def train_fft(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, logger):
    model = initialize_basic_model(num_classes, device)
    optimizer = optimizer(model.parameters(), lr=lr)
    logger.log_param_counts(model)
    print("\n=== Training with Full Fine-Tuning ===")
    for epoch in range(num_epochs):
        start = time.time()
        loss, acc = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, logger)
        val_acc = evaluate_model(model, val_loader, device)
        elapsed = time.time() - start
        logger.log_epoch_metrics(epoch, loss, acc, elapsed, torch.cuda.max_memory_allocated())
        print(f"FFT Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"FFT Test Acc: {test_acc:.4f}")
    model_path = f"saved_models/{logger.model_name}_{logger.timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

def train_metalr(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, hyper_lr, logger):
    model = initialize_basic_model(num_classes, device)
    meta_model = initialize_basic_model(num_classes, device)
    optimizer = optimizer(model.parameters(), lr=lr)
    logger.log_param_counts(model)
    print("\n=== Training with MetaLR ===")
    now_lr = [0.1 * lr] * 18 + [lr]
    train_meta_step.meta_iterator = iter(val_loader)
    for epoch in range(num_epochs):
        model.train()
        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            loss, _, now_lr = train_meta_step(images, labels, model, meta_model, now_lr, val_loader,
                                              optimizer, criterion, epoch, logger=logger,
                                              hyper_lr=hyper_lr, batch_idx=batch_idx, num_batches=num_batches)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"MetaLR Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"MetaLR Test Acc: {test_acc:.4f}")
    model_path = f"saved_models/{logger.model_name}_{logger.timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

def train_conv_adapters(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, reduction, lr, logger):
    model = initialize_conv_model(num_classes, device, reduction=reduction)
    optimizer = optimizer(model.parameters(), lr=lr)
    logger.log_param_counts(model)
    print("\n=== Training with Conv-Adapters ===")
    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, logger)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Conv Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Conv Test Acc: {test_acc:.4f}")
    model_path = f"saved_models/{logger.model_name}_{logger.timestamp}.pt"
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
    logger.log_param_counts(model)
    print("\n=== Training with Lora-C ===")
    for epoch in range(num_epochs):
        loss, acc = train_one_epoch(model, train_loader, optimizer, device, criterion, epoch, logger)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Lora-C Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"Lora-C Test Acc: {test_acc:.4f}")
    model_path = f"saved_models/{logger.model_name}_{logger.timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    logger.log_model_path(model_path)
    logger.save()

    
def train_models(run_fft, run_metalr, run_conv_adapters, run_lora, num_classes, train_loader, val_loader, test_loader, 
                 criterion, optimizer, device, num_epochs, lr, hyper_lr, reduction, r, lora_alpha):
    if run_fft:
        logger = make_logger("FFT")
        train_fft(num_classes, train_loader, val_loader, test_loader,
                  criterion, optimizer, device, num_epochs, lr=lr, logger=logger)
    if run_metalr:
        logger = make_logger("MetaLR")
        train_metalr(num_classes, train_loader, val_loader, test_loader,
                     criterion, optimizer, device, num_epochs, lr, hyper_lr, logger)
    if run_conv_adapters:
        logger = make_logger("ConvAdapter")
        train_conv_adapters(num_classes, train_loader, val_loader,
                            test_loader, criterion, optimizer,
                            device, num_epochs, reduction, lr=lr, logger=logger)
    if run_lora:
        logger = make_logger("LoRA-C")
        train_lora(num_classes, train_loader, val_loader,
                   test_loader, criterion, optimizer,
                   device, num_epochs, r=r, lora_alpha=lora_alpha, lr=lr, logger=logger)
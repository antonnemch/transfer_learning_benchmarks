
from models.conv_adapter_module import initialize_conv_model
from models.lora_layers import initialize_lora_model
from models.resnet_base import initialize_basic_model
from utils.meta_utils import train_meta_step
from utils.resnet_utils import train_one_epoch
from utils.resnet_utils import evaluate_model


def train_fft(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs):
    # Initialize models
    model = initialize_basic_model(num_classes, device)

    # Add save_parameters_by_module_csv once implemented

    print("\n=== Training with Full Fine-Tuning ===")
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"FFT Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device) 
    print(f"FFT Test Acc: {test_acc:.4f}")
    # Add save model to path once implemented


def train_metalr(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, lr, hyper_lr):
    # Initialize models
    model = initialize_basic_model(num_classes, device)
    meta_model = initialize_basic_model(num_classes, device)

    # Add save_parameters_by_module_csv once implemented

    print("\n=== Training with MetaLR ===")
    now_lr = [0.1 * lr] * 18 + [lr]
    train_meta_step.meta_iterator = iter(val_loader)
    for epoch in range(num_epochs):
        model.train()
        num_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            loss, _, now_lr = train_meta_step(
                images, labels, model, meta_model, now_lr, val_loader,
                optimizer, criterion, epoch,hyper_lr=hyper_lr,
                batch_idx=batch_idx, num_batches=num_batches
            )
        val_acc = evaluate_model(model, val_loader, device)
        print(f"MetaLR Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device)
    print(f"MetaLR Test Acc: {test_acc:.4f}")
    # Add save model and model results to path once implemented

def train_conv_adapters(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, reduction = 64):
    # Initialize models
    model = initialize_conv_model(num_classes, device, reduction=reduction)

    # Add save_parameters_by_module_csv once implemented

    # Criterion is unused in this function, but kept for consistency, uses Cross_entropy
    print("\n=== Training with Conv-Adapters ===")
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Conv Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device) 
    print(f"Conv Test Acc: {test_acc:.4f}")
    # Add save model and model results to path once implemented

def train_lora(num_classes, train_loader, val_loader, test_loader, criterion, optimizer, device, num_epochs, r = 16, lora_alpha = 32):
    
    custom_lora_config = {
    "r": r,
    "lora_alpha": lora_alpha,
    "lora_dropout": 0,
    "merge_weights": True
}
    # Initialize models
    model = initialize_lora_model(num_classes, device, lora_config=custom_lora_config)

    # Add save_parameters_by_module_csv once implemented

    # Criterion is unused in this function, but kept for consistency, uses Cross_entropy
    print("\n=== Training with Lora-C ===")
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, device, criterion)
        val_acc = evaluate_model(model, val_loader, device)
        print(f"Lora-C Epoch {epoch+1}/{num_epochs} - Val Acc: {val_acc:.4f}")
    test_acc = evaluate_model(model, test_loader, device) 
    print(f"Lora-C Test Acc: {test_acc:.4f}")
    # Add save model and model results to path once implemented

def train_models(train_fft, train_metalr, train_conv_adapters, train_lora,num_classes, train_loader, val_loader, test_loader, 
                 criterion, optimizer, device, num_epochs, lr, hyper_lr, reduction, r, lora_alpha):
    if train_fft:
        train_fft(num_classes, train_loader, val_loader, test_loader,
                  criterion, optimizer, device, num_epochs)
    if train_metalr:
        train_metalr(num_classes, train_loader, val_loader, test_loader,
                     criterion, optimizer, device, num_epochs, lr, hyper_lr)
    if train_conv_adapters:
        train_conv_adapters(num_classes, train_loader, val_loader,
                            test_loader, criterion, optimizer,
                            device, num_epochs, reduction)
    if train_lora:
        train_lora(num_classes, train_loader, val_loader,
                   test_loader, criterion, optimizer,
                   device, num_epochs, r=r, lora_alpha=lora_alpha)
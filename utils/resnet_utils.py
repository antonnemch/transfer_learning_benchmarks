import torch

def train_one_epoch(model, dataloader, optimizer, device, criterion, epoch, logger=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct_batch = (preds == labels).sum().item()
        total_batch = labels.size(0)

        total_loss += loss.item() * total_batch
        correct += correct_batch
        total += total_batch

        batch_acc = correct_batch / total_batch
        if logger:
            logger.log_batch_metrics(epoch, batch_idx, loss.item(), batch_acc)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def evaluate_model(model, dataloader, device, criterion=None, logger=None, epoch=None, phase="val"):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            # Compute loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)  # Sum loss over batch

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / total if criterion is not None else None

    class_names = dataloader.dataset.classes if hasattr(dataloader.dataset, 'classes') else None
    if logger is not None and class_names is not None:
        logger.log_confusion_matrix(y_true, y_pred, class_names, epoch=epoch if phase == "val" else None)

    return (avg_loss, acc) if criterion is not None else acc


# Prints the number of trainable and frozen parameters in a model,
# including breakdown by top-level module

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if p.requires_grad == False)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {frozen:,}")
    print(f"Trainable ratio:      {trainable / total:.2%}")
    print(f"Frozen ratio:         {frozen / total:.2%}")
    return total, trainable, frozen

def count_parameters_by_module(model):
    print("\n--- Parameter breakdown by module ---")
    for name, module in model.named_children():
        module_total = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_frozen = sum(p.numel() for p in module.parameters() if p.requires_grad == False)
        print(f"{name:20} | total: {module_total:,} | trainable: {module_trainable:,} | frozen: {module_frozen:,}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if self.best_val - val_loss > self.min_delta:
            self.best_val = val_loss
            self.counter = 0
            return False  # No early stop
        else:
            self.counter += 1
            return self.counter >= self.patience
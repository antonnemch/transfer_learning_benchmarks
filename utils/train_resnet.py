# A simple training loop for ResNet with ConvAdapters

import torch.nn.functional as F

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy


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

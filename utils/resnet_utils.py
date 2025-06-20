import torch
import torch.nn as nn
from models.custom_activations import AllActivations, KGActivation, LaplacianGPAF
from models.custom_activations import ChannelwiseActivation, CustomActivationPlaceholder

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
        print(f"Confusion matrix logged for {phase} phase.")

    return (avg_loss, acc) if phase != "test" else acc


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
        

def train_one_epoch_spline(model, dataloader, optimizers, device, criterion, epoch, logger=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        # dual optimizer setup
        for opt in optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in optimizers:
            opt.step()

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

def compute_num_experiments(model_name, run_models = {"GPAF":True}, hyperparams = None, model_param_map = None):
    # === Precompute total number of configurations across all models ===
    total_experiments = 0
    for model_name, should_run in run_models.items():
        if not should_run:
            continue
        relevant_params = model_param_map[model_name]
        param_counts = [len(hyperparams[p]) for p in relevant_params]
        model_total = 1
        for count in param_counts:
            model_total *= count
        total_experiments += model_total

    return total_experiments

def print_model_activations(model):
    for module_name, module in model.named_modules():
        for attr_name in dir(module):
            if attr_name.startswith('_'):
                continue  # skip private attributes

            attr_value = getattr(module, attr_name)

            if isinstance(attr_value, nn.Module):
                if isinstance(attr_value, AllActivations.ACTIVATION_TYPES):
                    if isinstance(attr_value, CustomActivationPlaceholder):
                        act_fn = attr_value.act_fn
                        if isinstance(act_fn, nn.Module):
                            print(f"{module_name}.{attr_name}: CustomActivationPlaceholder -> {act_fn}")
                            for param_name, param in act_fn.named_parameters(recurse=False):
                                print(f"  Param {param_name}: {param.detach().cpu().numpy()}")
                        else:
                            print(f"{module_name}.{attr_name}: CustomActivationPlaceholder -> [UNSET]")
                    elif isinstance(attr_value, ChannelwiseActivation):
                        print(f"{module_name}.{attr_name}: ChannelwiseActivation with {len(attr_value.activations)} activations")
                        for i, sub_act in enumerate(attr_value.activations):
                            print(f"  Channel {i}: {sub_act}")
                            for param_name, param in sub_act.named_parameters(recurse=False):
                                print(f"    Param {param_name}: {param.detach().cpu().numpy()}")
                    else:
                        print(f"{module_name}.{attr_name}: {attr_value}")
                        for param_name, param in attr_value.named_parameters(recurse=False):
                            print(f"  Param {param_name}: {param.detach().cpu().numpy()}")

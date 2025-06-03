# meta_train_utils.py
# Utilities for MetaLR training loop

import torch
from models.metaLR import MetaSGD
from models.resnet_base import resnet50_base

def get_optimizer(model, now_lr):
    return torch.optim.Adam([
        {'params': model.conv1.parameters(), 'lr': now_lr[0]},
        {'params': model.bn1.parameters(), 'lr': now_lr[1]},
        {'params': model.layer1[0].parameters(), 'lr': now_lr[2]},
        {'params': model.layer1[1].parameters(), 'lr': now_lr[3]},
        {'params': model.layer1[2].parameters(), 'lr': now_lr[4]},
        {'params': model.layer2[0].parameters(), 'lr': now_lr[5]},
        {'params': model.layer2[1].parameters(), 'lr': now_lr[6]},
        {'params': model.layer2[2].parameters(), 'lr': now_lr[7]},
        {'params': model.layer2[3].parameters(), 'lr': now_lr[8]},
        {'params': model.layer3[0].parameters(), 'lr': now_lr[9]},
        {'params': model.layer3[1].parameters(), 'lr': now_lr[10]},
        {'params': model.layer3[2].parameters(), 'lr': now_lr[11]},
        {'params': model.layer3[3].parameters(), 'lr': now_lr[12]},
        {'params': model.layer3[4].parameters(), 'lr': now_lr[13]},
        {'params': model.layer3[5].parameters(), 'lr': now_lr[14]},
        {'params': model.layer4[0].parameters(), 'lr': now_lr[15]},
        {'params': model.layer4[1].parameters(), 'lr': now_lr[16]},
        {'params': model.layer4[2].parameters(), 'lr': now_lr[17]},
        {'params': model.fc.parameters(), 'lr': now_lr[18]}
    ])


def train_meta_step(inputs, labels, model, meta_model, now_lr, meta_loader, optimizer, criterion, epoch, logger=None, hyper_lr=0.1, batch_idx=None, num_batches=None):
    total_loss = 0.0
    correct = 0
    total = 0

    # Step 1: Clone model into meta_model
    meta_model = resnet50_base(pretrained=False, num_classes=model.fc.out_features).to(inputs.device)
    meta_model.load_state_dict(model.state_dict())

    # Step 2: Forward pass on support set
    outputs = meta_model(inputs)
    loss_hat = criterion(outputs, labels)

    # Step 3: Compute gradients wrt model parameters
    meta_model.zero_grad()
    grads = torch.autograd.grad(loss_hat, meta_model.parameters(), create_graph=True)

    # Step 4: Construct differentiable learning rates tensor
    lrs = torch.tensor(now_lr, dtype=torch.float32, requires_grad=True)
    lrs.retain_grad()

    # Step 5: Simulate inner update
    pseudo_optimizer = MetaSGD(meta_model, meta_model.parameters(), lr=0.001)
    pseudo_optimizer.meta_step(grads, lrs)

    # Step 6: Get a batch from the validation set (meta set)
    try:
        meta_inputs, meta_labels = next(train_meta_step.meta_iterator)
    except StopIteration:
        train_meta_step.meta_iterator = iter(meta_loader)
        meta_inputs, meta_labels = next(train_meta_step.meta_iterator)

    meta_inputs, meta_labels = meta_inputs.to(inputs.device), meta_labels.to(inputs.device)

    # Step 7: Compute meta loss
    meta_outputs = meta_model(meta_inputs)
    loss_meta = criterion(meta_outputs, meta_labels)

    # Step 8: Backprop into lrs using meta loss
    lr_grads = torch.autograd.grad(loss_meta, lrs, create_graph=True)
    new_lr = lrs * (1 - lr_grads[0] * hyper_lr)
    new_lr = torch.clamp(new_lr, min=1e-6, max=1e-2).detach().tolist()

    # Step 9: Fix final layer learning rate for stability in early training
    if epoch < 5:
        new_lr[-1] = 0.01  # original base lr used in the MetaLR paper

    # Step 10: Apply updated learning rates
    s = optimizer.state_dict()
    for i, group in enumerate(s['param_groups']):
        group['lr'] = new_lr[i]
    optimizer.load_state_dict(s)

    # Step 11: Train main model on support set with updated LRs
    outputs = model(inputs)
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
        logger.log_metalr_lrs(epoch, batch_idx, new_lr)
        logger.log_batch_metrics(epoch, batch_idx, loss.item(), batch_acc, loss_meta.item())

    if batch_idx is not None and num_batches is not None:
        print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f} | MetaLoss: {loss_meta.item():.4f}")


    return loss, outputs, new_lr

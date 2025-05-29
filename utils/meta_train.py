# meta_train_utils.py
# Utilities for MetaLR training loop

import torch
from models.meta50 import MetaSGD
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

def train_meta_step(inputs, labels, model, meta_model, now_lr, meta_loader, optimizer, criterion, epoch, hyper_lr=0.1, batch_idx=None, num_batches=None): 
    meta_model = resnet50_base(pretrained=False, num_classes=model.fc.out_features).to(inputs.device)
    meta_model.load_state_dict(model.state_dict())

    outputs = meta_model(inputs)
    loss_hat = criterion(outputs, labels)

    meta_model.zero_grad()
    params = list(meta_model.parameters())
    grads = torch.autograd.grad(loss_hat, params, create_graph=True)

    lrs = torch.tensor(now_lr, dtype=torch.float64, requires_grad=True)

    params = [p.detach().clone().requires_grad_() for p in meta_model.parameters()]
    pseudo_optimizer = MetaSGD(meta_model, params, lr=0.001)
    pseudo_optimizer.meta_step(grads, lrs)

    try:
        meta_inputs, meta_labels = next(train_meta_step.meta_iterator)
    except StopIteration:
        train_meta_step.meta_iterator = iter(meta_loader)
        meta_inputs, meta_labels = next(train_meta_step.meta_iterator)

    meta_inputs, meta_labels = meta_inputs.to(inputs.device), meta_labels.to(inputs.device)
    meta_outputs = meta_model(meta_inputs)
    loss_meta = criterion(meta_outputs, meta_labels)

    lr_grads = torch.autograd.grad(loss_meta, lrs, create_graph=True)
    new_lr = lrs * (1 - lr_grads[0] * hyper_lr)
    new_lr = torch.clamp(new_lr, min=1e-6, max=1e-2).detach().tolist()

    if epoch <= 5:
        new_lr[-1] = now_lr[-1]

    s = optimizer.state_dict()
    for i, group in enumerate(s['param_groups']):
        group['lr'] = new_lr[i]
    optimizer.load_state_dict(s)

    outputs = model(inputs)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx is not None and num_batches is not None:
        print(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f} | MetaLoss: {loss_meta.item():.4f}")

    return loss, outputs, new_lr

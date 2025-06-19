import os
import yaml
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm

# Import dataset utilities
from dataset.CUB200 import create_cub_dataloaders
from dataset.IP102 import create_ip102_dataloaders
from dataset.StandfordDogs import StanfordDogsDataset, get_transforms


def get_dataloaders(cfg, model=None):
    name = cfg['dataset']['name']
    root = cfg['dataset']['root']
    batch_size = cfg['dataset']['batch_size']
    num_workers = cfg['dataset']['num_workers']
    
    # Get timm transforms if model is provided
    if model is not None:
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        test_transform = timm.data.create_transform(**data_cfg)
        # For training, add data augmentation
        train_data_cfg = data_cfg.copy()
        train_data_cfg.update({'is_training': True})
        train_transform = timm.data.create_transform(**train_data_cfg)
    else:
        # Fallback to default transforms
        train_transform = test_transform = None
    
    if name == 'CUB200':
        if train_transform is not None:
            # Create custom dataloaders with timm transforms
            from dataset.CUB200 import CUB200Dataset
            train_ds = CUB200Dataset(root, train=True, transform=train_transform, download=True)
            test_ds = CUB200Dataset(root, train=False, transform=test_transform, download=False)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            train_loader, test_loader = create_cub_dataloaders(root, batch_size, num_workers)
    elif name == 'IP102':
        if train_transform is not None:
            # Create custom dataloaders with timm transforms
            from dataset.IP102 import IP102Dataset
            train_ds = IP102Dataset(root, split='train', transform=train_transform, download=True)
            test_ds = IP102Dataset(root, split='test', transform=test_transform, download=False)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            dl = create_ip102_dataloaders(root, batch_size, num_workers, download=True)
            train_loader, test_loader = dl['train'], dl['test']
    elif name == 'StanfordDogs':
        if train_transform is not None:
            train_ds = StanfordDogsDataset(root, train=True, transform=train_transform, download=True)
            test_ds = StanfordDogsDataset(root, train=False, transform=test_transform, download=False)
        else:
            train_ds = StanfordDogsDataset(root, train=True, transform=get_transforms(True), download=True)
            test_ds = StanfordDogsDataset(root, train=False, transform=get_transforms(False), download=False)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        raise ValueError(f"Unsupported dataset {name}")
    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='Training', leave=False)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)
        pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    return running_loss/total, correct/total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    pbar = tqdm(loader, desc='Evaluating', leave=False)
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += inputs.size(0)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
    return running_loss/total, correct/total, y_true, y_pred


def main():
    # Load configuration
    cfg_path = os.path.join(os.path.dirname(__file__), 'main.yaml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Prepare logging
    save_dir = cfg['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Device
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')

    # Model (create first to get transforms)
    model_name = cfg['model']['name']
    pretrained = cfg['model'].get('pretrained', False)
    # First create with default num_classes to get pretrained_cfg
    temp_model = timm.create_model(model_name, pretrained=pretrained)
    
    # Data (now with model-specific transforms)
    train_loader, test_loader = get_dataloaders(cfg, temp_model)
    num_classes = len(train_loader.dataset.classes) if hasattr(train_loader.dataset, 'classes') else len(test_loader.dataset.classes)
    
    # Create final model with correct num_classes
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])

    # Training loop
    epochs = cfg['training']['epochs']
    history = []
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

    # Save metrics
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    # Save confusion matrix of last epoch
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

if __name__ == '__main__':
    main()
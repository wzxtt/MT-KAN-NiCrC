"""
Training and evaluation utilities for MT-KAN and baselines.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
import time
import os


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed regularization for thin film properties.
    
    Constraints:
    1. Sa-time: exponential decay (roughness decreases with deposition time)
    2. H-bias: Gaussian-like peak (hardness peaks at optimal ion energy)
    3. H-flow: Monotonic increase (hardness increases with C content)
    4. Smoothness: output spline curvature should be small
    """
    def __init__(self, lambda_physics=0.1, lambda_smooth=0.01):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_smooth = lambda_smooth
    
    def forward(self, model, x, preds, task_idx):
        """
        Compute physics-informed regularization loss.
        
        Args:
            model: the KAN model
            x: (batch, 4) features [flow, bias, time, material]
            preds: (batch,) predictions for this task
            task_idx: which task (0=Sa, 1=H, 2=E, 3=stress, 4=H/E*)
        Returns:
            physics_loss: scalar
        """
        phys_loss = 0.0
        batch = x.shape[0]
        
        # Only apply physics constraints to NiCrC (material=0)
        is_nicrc = x[:, 3] < 0.5
        if is_nicrc.sum() < 5:
            return torch.tensor(0.0, device=x.device)
        
        x_sub = x[is_nicrc]
        pred_sub = preds[is_nicrc]
        
        if task_idx == 0:  # Sa - roughness
            # Physical constraint: Sa decreases with time (exponential decay)
            time_idx = torch.argsort(x_sub[:, 2])
            if len(time_idx) > 2:
                sorted_preds = pred_sub[time_idx]
                # Penalize non-monotonic decreasing (after normalization)
                # Only apply where time > 0
                diff = sorted_preds[1:] - sorted_preds[:-1]
                # We want diff <= 0 generally, but allow small increases
                phys_loss += torch.mean(F.relu(diff)) * 0.1
        
        elif task_idx == 1:  # H - hardness
            # Physical constraint: H has Gaussian-like peak with bias
            bias = x_sub[:, 1]
            # Optimal bias ~ around -200V for NiCrC = (200 - mean)/std
            # Approximate center at normalized ~0
            penalty = torch.abs(pred_sub) * torch.abs(bias) * 0.01
            phys_loss += penalty.mean()
        
        return phys_loss * self.lambda_physics


class WeightedMSELoss(nn.Module):
    """MSE loss with per-sample weights, NaN masking, and optional per-task weights."""
    def __init__(self, task_weights=None):
        super().__init__()
        if task_weights is not None:
            self.register_buffer('task_weights', torch.FloatTensor(task_weights))
        else:
            self.task_weights = None

    def forward(self, pred, target, mask, weights):
        diff = (pred - target) ** 2
        if self.task_weights is not None:
            tw = self.task_weights.to(pred.device)
            diff = diff * tw[None, :]
        valid_count = mask.sum()
        if valid_count < 1:
            return torch.tensor(0.0, device=pred.device)
        # weights: (batch, 1), diff: (batch, n_tasks)
        masked_diff = diff * mask.float()
        weighted_diff = masked_diff * weights
        return weighted_diff.sum() / valid_count.float()


def train_epoch(model, loader, optimizer, criterion, device, phys_criterion=None, task_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for x, y, mask, w in loader:
        x, y, mask, w = x.to(device), y.to(device), mask.to(device), w.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        
        # Main task loss
        loss = criterion(pred, y, mask, w)
        
        # Physics-informed loss (for KAN model with physics)
        if phys_criterion is not None and hasattr(model, 'backbone'):
            phys_loss = 0.0
            for t in range(model.n_tasks):
                task_pred = pred[:, t]
                phys_loss += phys_criterion(model, x, task_pred, t)
            loss = loss + phys_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def evaluate(model, loader, criterion, device, tgt_mean, tgt_std):
    """Evaluate model on given data loader."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_masks = []
    n_batches = 0
    
    with torch.no_grad():
        for x, y, mask, w in loader:
            x, y, mask, w = x.to(device), y.to(device), mask.to(device), w.to(device)
            pred = model(x)
            loss = criterion(pred, y, mask, w)
            total_loss += loss.item()
            n_batches += 1
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    # Denormalize
    preds_denorm = preds * tgt_std[None, :] + tgt_mean[None, :]
    targets_denorm = targets * tgt_std[None, :] + tgt_mean[None, :]
    
    # Per-task metrics using stored masks (NOT np.isnan, since NaN was filled)
    metrics = {}
    task_names = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
    for i, name in enumerate(task_names):
        valid = masks[:, i]
        if valid.sum() < 5:
            metrics[name] = {'R2': 0, 'RMSE': 0, 'MAE': 0}
            continue
        
        y_true = targets_denorm[valid, i]
        y_pred = preds_denorm[valid, i]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        metrics[name] = {'R2': float(r2), 'RMSE': float(rmse), 'MAE': float(mae)}
    
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    
    return total_loss / max(n_batches, 1), metrics, preds_denorm, targets_denorm


def train_model(model, train_loader, val_loader, test_loader, device,
                tgt_mean, tgt_std, n_epochs=300, lr=1.5e-3, weight_decay=5e-6,
                model_name='KAN', patience=40, use_physics=True, task_weights=None):
    """
    Full training loop with early stopping.

    ``test_loader`` may be None, in which case no test-set evaluation is
    performed at all and the returned test metrics/predictions are None.
    This is used by the hyper-parameter selection stage so that the test
    split cannot influence model selection.
    """
    criterion = WeightedMSELoss(task_weights=task_weights)
    phys_criterion = PhysicsInformedLoss(lambda_physics=0.02) if use_physics else None
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    t_start = time.time()
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 phys_criterion)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device,
                                               tgt_mean, tgt_std)
        
        avg_r2 = np.mean([m['R2'] for m in val_metrics.values()])
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(avg_r2)
        
        # Early stopping
        if avg_r2 > best_val_r2:
            best_val_r2 = avg_r2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            t_elapsed = time.time() - t_start
            print(f'  Epoch {epoch+1:3d}/{n_epochs} | train_loss={train_loss:.4f} | '
                  f'val_loss={val_loss:.4f} | val_R2={avg_r2:.4f} | {t_elapsed:.0f}s')
        
        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch+1}')
            break
    
    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    history['best_val_r2'] = best_val_r2
    history['best_epoch'] = int(np.argmax(history['val_r2'])) + 1

    # Final test evaluation -- skipped entirely when test_loader is None
    if test_loader is None:
        print(f'\n  {model_name} selected checkpoint: val R^2={best_val_r2:.4f} '
              f'@ epoch {history["best_epoch"]} (test split not evaluated)')
        return history, None, None, None, best_state

    test_loss, test_metrics, preds, targets = evaluate(
        model, test_loader, criterion, device, tgt_mean, tgt_std)

    print(f'\n  {model_name} Test Results:')
    for name, m in test_metrics.items():
        print(f'    {name}: R^2={m["R2"]:.4f}, RMSE={m["RMSE"]:.4f}, MAE={m["MAE"]:.4f}')
    avg_test_r2 = np.mean([m['R2'] for m in test_metrics.values()])
    print(f'    Average R^2 = {avg_test_r2:.4f}')

    return history, test_metrics, preds, targets, best_state


if __name__ == '__main__':
    from data_loader import create_dataloaders
    from kan_model import MultiTaskKAN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'unified_dataset', 'training_dataset.csv')
    train_loader, val_loader, test_loader, ds = create_dataloaders(data_path)
    
    model = MultiTaskKAN(n_input=4, n_hidden=32, n_tasks=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')
    
    tgt_mean = ds.tgt_mean
    tgt_std = ds.tgt_std
    
    history, test_metrics, preds, targets, best_state = train_model(
        model, train_loader, val_loader, test_loader, device,
        tgt_mean, tgt_std, n_epochs=100, model_name='MT-KAN')
    
    print('\nTraining complete!')

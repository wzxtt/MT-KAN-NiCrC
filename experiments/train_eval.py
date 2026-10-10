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
    Sweep-based physics-informed regularisation for thin-film properties.

    The constraints are evaluated by differentiably forward-passing a fixed
    context sweep instead of in-batch samples (whose deposition times are
    mostly mean-imputed, which made the old in-batch penalty a noisy signal):

    1. Sa-time: predicted roughness must decrease with deposition time
       (NiCrC data: r = -0.94 between Sa and log time).
    2. H-bias: predicted hardness at extreme bias must not exceed the
       hardness near the optimal bias (measured peak at 300-400 V,
       16.9 GPa, versus 8.6-9.5 GPa at the bias extremes).

    Small hinge margins (delta_sa, delta_h) leave numerically flat regions
    unpunished. set_grids() must be called with the dataset feature
    normalisation and raw NiCrC context rows before training starts.
    """

    def __init__(self, lambda_physics=0.5, delta_sa=0.02, delta_h=0.05,
                 n_ctx=8, n_time=32, n_bias=32, n_extreme=8):
        super().__init__()
        self.lambda_physics = float(lambda_physics)
        self.delta_sa = float(delta_sa)
        self.delta_h = float(delta_h)
        self.n_ctx = int(n_ctx)
        self.n_time = int(n_time)
        self.n_bias = int(n_bias)
        self.n_extreme = int(n_extreme)
        self.grids_ready = False

    def set_grids(self, feat_mean, feat_std, contexts):
        """contexts: raw (n, 4) NiCrC rows [flow, bias, time, material]."""
        mean = np.asarray(feat_mean, dtype=np.float64)
        std = np.asarray(feat_std, dtype=np.float64)
        ctx = np.asarray(contexts, dtype=np.float64)[: self.n_ctx].copy()
        bad = ~np.isfinite(ctx)
        if bad.any():
            ctx[bad] = np.broadcast_to(mean, ctx.shape)[bad]

        time_n = (np.linspace(10.0, 3600.0, self.n_time) - mean[2]) / std[2]
        bias_n = (np.linspace(0.0, 800.0, self.n_bias) - mean[1]) / std[1]
        ctx_n = (ctx - mean[None, :]) / std[None, :]

        n, m = self.n_ctx, self.n_time
        T = np.zeros((n * m, 4))
        T[:, 0] = np.repeat(ctx_n[:, 0], m)
        T[:, 1] = np.repeat(ctx_n[:, 1], m)
        T[:, 2] = np.tile(time_n, n)
        T[:, 3] = np.repeat(ctx_n[:, 3], m)

        mb = self.n_bias
        B = np.zeros((n * mb, 4))
        B[:, 0] = np.repeat(ctx_n[:, 0], mb)
        B[:, 1] = np.tile(bias_n, n)
        B[:, 2] = np.repeat(ctx_n[:, 2], mb)
        B[:, 3] = np.repeat(ctx_n[:, 3], mb)

        C = ctx_n.copy()
        C[:, 1] = 0.0                       # centre probe: normalised bias 0

        extreme = np.sort(np.argsort(-np.abs(bias_n))[: self.n_extreme])

        self.register_buffer('T', torch.from_numpy(T.astype(np.float32)))
        self.register_buffer('B', torch.from_numpy(B.astype(np.float32)))
        self.register_buffer('C', torch.from_numpy(C.astype(np.float32)))
        self.register_buffer('extreme_idx', torch.from_numpy(extreme.astype(np.int64)))
        self.grids_ready = True

    def forward(self, model, x, preds, task_idx):
        if not self.grids_ready:
            return preds.new_zeros(())
        if task_idx == 0:
            sa = model(self.T)[:, 0].view(self.n_ctx, self.n_time)
            diff = sa[:, 1:] - sa[:, :-1]
            return F.relu(diff - self.delta_sa).mean() * self.lambda_physics
        if task_idx == 1:
            h = model(self.B)[:, 1].view(self.n_ctx, self.n_bias)
            h_c = model(self.C)[:, 1]
            h_ext = h.index_select(1, self.extreme_idx)
            return F.relu(h_ext - h_c.unsqueeze(1) - self.delta_h).mean() * self.lambda_physics
        return preds.new_zeros(())


def sample_phys_contexts(ds, n_ctx=8, seed=1):
    """Raw NiCrC rows [flow, bias, time, material] used to anchor sweeps."""
    ni = ds.df[(ds.df['material'] == 0.0) & ds.df['time_s'].notna()]
    rs = np.random.RandomState(seed)
    pos = rs.choice(len(ni), size=n_ctx, replace=False)
    cols = ['C2H2_sccm', 'bias_V', 'time_s', 'material']
    return ni.iloc[pos][cols].to_numpy(dtype=np.float64)


def build_consistency_grids(ds, n_ctx=32, n_time=60, n_bias=60,
                            n_extreme=12, seed=0):
    """Fixed sweep grids used to score physical consistency (eval side)."""
    ni = ds.df[(ds.df['material'] == 0.0) & ds.df['time_s'].notna()]
    rs = np.random.RandomState(seed)
    pos = rs.choice(len(ni), size=n_ctx, replace=False)
    cols = ['C2H2_sccm', 'bias_V', 'time_s', 'material']
    ctx_raw = ni.iloc[pos][cols].to_numpy(dtype=np.float64)
    bad = ~np.isfinite(ctx_raw)
    if bad.any():
        ctx_raw[bad] = np.broadcast_to(ds.feat_mean, ctx_raw.shape)[bad]

    time_n = (np.linspace(10.0, 3600.0, n_time) - ds.feat_mean[2]) / ds.feat_std[2]
    bias_n = (np.linspace(0.0, 800.0, n_bias) - ds.feat_mean[1]) / ds.feat_std[1]
    ctx_n = (ctx_raw - ds.feat_mean[None, :]) / ds.feat_std[None, :]

    T = np.zeros((n_ctx * n_time, 4))
    T[:, 0] = np.repeat(ctx_n[:, 0], n_time)
    T[:, 1] = np.repeat(ctx_n[:, 1], n_time)
    T[:, 2] = np.tile(time_n, n_ctx)
    T[:, 3] = np.repeat(ctx_n[:, 3], n_time)

    B = np.zeros((n_ctx * n_bias, 4))
    B[:, 0] = np.repeat(ctx_n[:, 0], n_bias)
    B[:, 1] = np.tile(bias_n, n_ctx)
    B[:, 2] = np.repeat(ctx_n[:, 2], n_bias)
    B[:, 3] = np.repeat(ctx_n[:, 3], n_bias)

    C = ctx_n.copy()
    C[:, 1] = 0.0
    extreme = np.sort(np.argsort(-np.abs(bias_n))[:n_extreme])
    return {'T': T.astype(np.float32), 'B': B.astype(np.float32),
            'C': C.astype(np.float32), 'extreme_idx': extreme,
            'n_ctx': n_ctx, 'n_time': n_time, 'n_bias': n_bias}


@torch.no_grad()
def consistency_metrics(model, grids, delta_sa=0.02, delta_h=0.05):
    """Sa-time monotonicity and H-bias extremum violation metrics.

    Strict rates count any increase / extremum excess; `_m` rates count only
    violations larger than the physical tolerance (the same margins used by
    the training hinge). `sa_pen` / `h_pen` are mean excess magnitudes.
    """
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device

    def pred(a):
        return model(torch.from_numpy(a).to(dev)).cpu().numpy()

    n_ctx, n_time, n_bias = grids['n_ctx'], grids['n_time'], grids['n_bias']
    sa = pred(grids['T']).reshape(n_ctx, n_time, -1)[:, :, 0]
    diff = sa[:, 1:] - sa[:, :-1]
    sa_viol = float((diff > 0).mean())
    sa_viol_m = float((diff > delta_sa).mean())
    sa_pen = float(np.maximum(diff, 0).mean())

    h = pred(grids['B']).reshape(n_ctx, n_bias, -1)[:, :, 1]
    h_c = pred(grids['C'])[:, 1]
    h_ext = h[:, grids['extreme_idx']]
    h_excess = h_ext - h_c[:, None]
    h_viol = float((h_excess > 0).mean())
    h_viol_m = float((h_excess > delta_h).mean())
    h_pen = float(np.maximum(h_excess, 0).mean())

    if was_training:
        model.train()
    return {'sa_viol': sa_viol, 'sa_viol_m': sa_viol_m, 'sa_pen': sa_pen,
            'h_viol': h_viol, 'h_viol_m': h_viol_m, 'h_pen': h_pen}


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


def train_epoch(model, loader, optimizer, criterion, device, phys_criterion=None,
                task_weights=None, phys_every=1):
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
        
        # Physics-informed loss on a fixed sweep grid, applied every
        # phys_every-th batch (the grid forward pass is much larger than a
        # batch, so frequent application would dominate training cost).
        if (phys_criterion is not None and hasattr(model, 'backbone')
                and n_batches % max(phys_every, 1) == 0):
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
                model_name='KAN', patience=40, use_physics=True, task_weights=None,
                lambda_physics=0.05, feat_mean=None, feat_std=None,
                phys_contexts=None, phys_every=4):
    """
    Full training loop with early stopping.

    ``test_loader`` may be None, in which case no test-set evaluation is
    performed at all and the returned test metrics/predictions are None.
    This is used by the hyper-parameter selection stage so that the test
    split cannot influence model selection.

    When ``use_physics`` is set, ``feat_mean``/``feat_std``/``phys_contexts``
    (raw NiCrC context rows) are required to build the sweep-based physics
    penalty; the penalty is applied every ``phys_every``-th batch.
    """
    criterion = WeightedMSELoss(task_weights=task_weights)
    phys_criterion = None
    if use_physics:
        if feat_mean is None or feat_std is None or phys_contexts is None:
            raise ValueError(
                'use_physics=True requires feat_mean, feat_std and phys_contexts '
                '(call train_eval.sample_phys_contexts(ds))')
        phys_criterion = PhysicsInformedLoss(lambda_physics=lambda_physics)
        phys_criterion.set_grids(feat_mean, feat_std, phys_contexts)
        phys_criterion = phys_criterion.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    t_start = time.time()
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                 phys_criterion, phys_every=phys_every)
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
    
    data_path = r'C:\Users\28100\Desktop\双因素影响\unified_dataset\training_dataset.csv'
    train_loader, val_loader, test_loader, ds = create_dataloaders(data_path)
    
    model = MultiTaskKAN(n_input=4, n_hidden=32, n_tasks=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {n_params:,}')
    
    tgt_mean = ds.tgt_mean
    tgt_std = ds.tgt_std
    
    history, test_metrics, preds, targets, best_state = train_model(
        model, train_loader, val_loader, test_loader, device,
        tgt_mean, tgt_std, n_epochs=100, model_name='MT-KAN',
        feat_mean=ds.feat_mean, feat_std=ds.feat_std,
        phys_contexts=sample_phys_contexts(ds))
    
    print('\nTraining complete!')

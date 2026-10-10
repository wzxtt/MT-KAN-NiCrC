"""
Main experiment script: MT-KAN for NiCrC thin film property prediction.
Trains KAN, baselines, generates all figures.
"""
import os, sys, json, time, warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

from data_loader import ThinFilmDataset
from kan_model import MultiTaskKAN, MLPBaseline
from train_eval import train_model, evaluate, WeightedMSELoss, sample_phys_contexts
from baselines import (
    train_xgboost_baseline, train_random_forest_baseline,
    train_svr_baseline, train_knn_baseline, train_elasticnet_baseline
)
from figures import generate_all_figures

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join(BASE_DIR, 'unified_dataset', 'training_dataset.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'results')
FIG_DIR = os.path.join(BASE_DIR, 'experiments', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TASK_NAMES = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
FEAT_NAMES = ['C2H2_flow', 'bias_voltage', 'dep_time', 'material']

SEED = 42
N_EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 40
# Task weights: boost hard tasks H, sigma, H/E*
TASK_WEIGHTS = [1.0, 2.5, 1.0, 2.5, 2.0]

np.random.seed(SEED)
torch.manual_seed(SEED)

_eval_criterion = WeightedMSELoss()

print('='*60)
print('MT-KAN Experiment: NiCrC Thin Film Property Prediction')
print('='*60)
print(f'Device: {DEVICE}')
print(f'Data: {DATA_PATH}')

# ============================================================
# 1. LOAD DATA & CREATE CONSISTENT SPLIT
# ============================================================
print('\n[1] Loading data...')
ds = ThinFilmDataset(DATA_PATH)
phys_ctx = sample_phys_contexts(ds)
all_X = ds.features
all_Y = ds.targets * ds.tgt_std[None, :] + ds.tgt_mean[None, :]  # denorm (NaN filled with mean)
all_Y_norm = ds.targets
all_mask = ds.valid_mask
all_weights = ds.weights

# Raw targets preserving NaN for tree-based models
raw_targets = ds.df[TASK_NAMES].values.astype(np.float32)
raw_targets_denorm = np.where(all_mask, all_Y, np.nan)  # NaN in original positions

# Single train/test split for all models
idx = np.arange(len(ds))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=SEED) # 0.125*0.8=0.1

print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

# Create DataLoaders for PyTorch models
train_ds = Subset(ds, train_idx)
val_ds = Subset(ds, val_idx)
test_ds = Subset(ds, test_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# Data for sklearn baselines
X_train, y_train = all_X[train_idx], all_Y[train_idx]
X_val, y_val = all_X[val_idx], all_Y[val_idx]
X_test, y_test = all_X[test_idx], all_Y[test_idx]
mask_test = all_mask[test_idx]

# Raw targets preserving NaN for tree-based models
y_train_raw = raw_targets_denorm[train_idx]
y_test_raw = raw_targets_denorm[test_idx]

# ============================================================
# 2. TRAIN MT-KAN
# ============================================================
print('\n[2] Training MT-KAN...')
kan_model = MultiTaskKAN(n_input=4, n_hidden=64, n_tasks=5).to(DEVICE)
n_params_kan = sum(p.numel() for p in kan_model.parameters())
print(f'  MT-KAN parameters: {n_params_kan:,}')

kan_history, kan_metrics, kan_preds, kan_targets, kan_best = train_model(
    kan_model, train_loader, val_loader, test_loader, DEVICE,
    ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS, model_name='MT-KAN',
    patience=PATIENCE, use_physics=True, task_weights=TASK_WEIGHTS,
    feat_mean=ds.feat_mean, feat_std=ds.feat_std,
    phys_contexts=phys_ctx)

test_loss, kan_metrics_full, kan_preds_full, kan_targets_full = evaluate(
    kan_model, test_loader, _eval_criterion, DEVICE, ds.tgt_mean, ds.tgt_std)

torch.save(kan_best, os.path.join(OUT_DIR, 'mt_kan_best.pth'))

# ============================================================
# 3. TRAIN MLP BASELINE
# ============================================================
print('\n[3] Training MLP baseline...')
mlp_model = MLPBaseline(n_input=4, n_hidden=64, n_tasks=5).to(DEVICE)
n_params_mlp = sum(p.numel() for p in mlp_model.parameters())
print(f'  MLP parameters: {n_params_mlp:,}')

mlp_history, mlp_metrics, mlp_preds, mlp_targets, mlp_best = train_model(
    mlp_model, train_loader, val_loader, test_loader, DEVICE,
    ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS, model_name='MLP',
    patience=PATIENCE, use_physics=False)

test_loss, mlp_metrics_full, mlp_preds_full, mlp_targets_full = evaluate(
    mlp_model, test_loader, _eval_criterion, DEVICE, ds.tgt_mean, ds.tgt_std)

# ============================================================
# 4-7. TRAIN SKLEARN BASELINES
# ============================================================
all_preds_dict = {}
all_metrics = {}

baseline_fns = [
    ('Random Forest', train_random_forest_baseline),
    ('XGBoost', train_xgboost_baseline),
    ('SVR', train_svr_baseline),
    ('kNN', train_knn_baseline),
    ('ElasticNet', train_elasticnet_baseline),
]

for name, fn in baseline_fns:
    print(f'\n[4] Training {name} baseline...')
    results, preds = fn(X_train, y_train_raw, X_test, y_test_raw, TASK_NAMES)
    all_metrics[name] = {k: v for k, v in results.items() if k in TASK_NAMES}

    preds_full = np.full_like(y_test, np.nan)
    for i, t in enumerate(TASK_NAMES):
        if t in results and 'model' in results[t]:
            valid_test = ~np.isnan(y_test_raw[:, i])
            if valid_test.sum() > 0:
                preds_full[valid_test, i] = results[t]['model'].predict(X_test[valid_test])
    all_preds_dict[name] = preds_full

# ============================================================
# 8. ABLATION: MT-KAN without physics
# ============================================================
print('\n[8] Ablation: MT-KAN w/o physics...')
kan_nophys = MultiTaskKAN(n_input=4, n_hidden=64, n_tasks=5).to(DEVICE)
kan_nophys_history, kan_nophys_metrics, _, _, _ = train_model(
    kan_nophys, train_loader, val_loader, test_loader, DEVICE,
    ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS, model_name='KAN (no physics)',
    patience=PATIENCE, use_physics=False)

# ============================================================
# 9. COMPILE RESULTS
# ============================================================
print('\n[9] Compiling results...')

all_metrics = {
    'MT-KAN': kan_metrics_full,
    'MLP': mlp_metrics_full,
    **all_metrics,
}

all_preds_dict = {
    'MT-KAN': kan_preds_full,
    'MLP': mlp_preds_full,
    **all_preds_dict,
}

history_dict = {'MT-KAN': kan_history, 'MLP': mlp_history,
                'KAN (no physics)': kan_nophys_history}

# Source labels for figures (from test set)
src_test = ds.df['source'].values[test_idx]
source_labels = {}
for src in ['exact', 'synthetic', 'digitized']:
    mask = np.array([s == src for s in src_test])
    if mask.sum() > 0:
        source_labels[src] = mask

ablation_metrics = {
    'MT-KAN (full)': kan_metrics_full,
    'w/o physics': kan_nophys_metrics,
    'MLP': mlp_metrics_full,
}

model_names = list(all_metrics.keys())

print('\n--- Performance Summary (R^2) ---')
for name in model_names:
    metrics = all_metrics[name]
    r2s = [metrics[t]['R2'] if t in metrics else 0 for t in TASK_NAMES]
    print(f'  {name:20s}: {", ".join([f"{r:.3f}" for r in r2s])} | avg={np.mean(r2s):.3f}')

results_summary = {}
for name, metrics in all_metrics.items():
    results_summary[name] = {}
    for t in TASK_NAMES:
        if t in metrics:
            results_summary[name][t] = {k: v for k, v in metrics[t].items() if k != 'model'}
results_summary['ablation'] = {}
for name, metrics in ablation_metrics.items():
    results_summary['ablation'][name] = {}
    for t in TASK_NAMES:
        if t in metrics:
            results_summary['ablation'][name][t] = {k: v for k, v in metrics[t].items() if k != 'model'}

with open(os.path.join(OUT_DIR, 'results_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print(f'\nResults saved to {OUT_DIR}/results_summary.json')

# ============================================================
# 10. GENERATE FIGURES
# ============================================================
print('\n[10] Generating figures...')
generate_all_figures(
    all_metrics=all_metrics,
    all_preds_dict=all_preds_dict,
    all_targets=y_test,
    history_dict=history_dict,
    model_names=model_names,
    task_names=TASK_NAMES,
    feature_names=FEAT_NAMES,
    model=kan_model,
    ds=ds,
    source_labels=source_labels,
    ablation_metrics=ablation_metrics,
    save_dir=FIG_DIR,
    mask_test=mask_test,
    X_test=X_test,
)

print('\n' + '='*60)
print('Experiment complete!')
print(f'Results: {OUT_DIR}/')
print(f'Figures: {FIG_DIR}/')
print('='*60)

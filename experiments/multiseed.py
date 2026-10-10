"""
Stage 3: multi-seed stability check (R2 + RMSE + MAE, all seven models).

Question
--------
The single-seed final run gave MT-KAN an average test R^2 of 0.645 against
0.617 for Random Forest. A gap of 0.028 is small, and a second MLP run on a
different partition moved the stress task by 0.13. Before those numbers go into
a manuscript we need to know whether the ranking survives resampling.

Protocol
--------
Five seeds. Each seed changes BOTH the train/val/test partition AND the network
initialisation, because the partition is the dominant variance source.

Per seed we fit MT-KAN (physics prior + frozen task weights), MT-KAN (neither),
the capacity-matched MLP, and the five sklearn baselines (RF, XGBoost, kNN,
SVR, ElasticNet) on the same split.

Comparison is PAIRED BY SEED: the MT-KAN minus baseline difference is taken
within each seed, which removes the split-to-split shift that otherwise
dominates the between-model spread.

No figures, no checkpoints - metrics only.
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

from data_loader import ThinFilmDataset
from kan_model import MultiTaskKAN, MLPBaseline
from train_eval import (train_model, evaluate, WeightedMSELoss,
                        sample_phys_contexts, build_consistency_grids,
                        consistency_metrics)
from baselines import (
    train_xgboost_baseline, train_random_forest_baseline,
    train_svr_baseline, train_knn_baseline, train_elasticnet_baseline
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join(BASE_DIR, 'unified_dataset', 'training_dataset.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'results')

TASK_NAMES = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
TASK_LABEL = {'Sa_nm': 'Sa', 'H_GPa': 'H', 'E_GPa': 'E',
              'stress_GPa': 'sigma', 'H_Estar': 'H/E*'}

METRICS = ('R2', 'RMSE', 'MAE')

SEEDS = [2026, 2027, 2028, 2029, 2030]
N_EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 40

SK_MODELS = [
    ('Random Forest', train_random_forest_baseline),
    ('XGBoost', train_xgboost_baseline),
    ('kNN', train_knn_baseline),
    ('SVR', train_svr_baseline),
    ('ElasticNet', train_elasticnet_baseline),
]
NEURAL = ['MT-KAN (full)', 'MT-KAN (neither)', 'MLP']


def load_frozen_weights():
    with open(os.path.join(OUT_DIR, 'task_weights.json')) as f:
        payload = json.load(f)
    return payload['selected_weights']


def r2_of(res, m, task=None):
    keys = TASK_NAMES if task is None else [task]
    return float(np.mean([res[m][k]['R2'] for k in keys]))


def run_one_seed(ds, seed, tw):
    """Fit every model on one partition. Returns {model: {task: {R2,RMSE,MAE}}}."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_X = ds.features
    all_Y_norm = ds.targets
    all_Y = all_Y_norm * ds.tgt_std[None, :] + ds.tgt_mean[None, :]
    all_mask = ds.valid_mask
    raw_denorm = np.where(all_mask, all_Y, np.nan)

    idx = np.arange(len(ds))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=seed)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE)

    X_train, X_test = all_X[train_idx], all_X[test_idx]
    y_train_raw = raw_denorm[train_idx]
    y_test_raw = raw_denorm[test_idx]

    eval_criterion = WeightedMSELoss()
    out = {}

    phys_ctx = sample_phys_contexts(ds)
    grids = build_consistency_grids(ds)

    configs = [
        ('MT-KAN (full)',    MultiTaskKAN, True,  tw),
        ('MT-KAN (neither)', MultiTaskKAN, False, None),
        ('MLP',              MLPBaseline, False, None),
    ]
    for name, ctor, use_phys, cell_tw in configs:
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = ctor(n_input=4, n_tasks=5).to(DEVICE)
        train_model(
            model, train_loader, val_loader, test_loader, DEVICE,
            ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS, model_name=name,
            patience=PATIENCE, use_physics=use_phys, task_weights=cell_tw,
            feat_mean=ds.feat_mean, feat_std=ds.feat_std,
            phys_contexts=phys_ctx)
        _, metrics, _, _ = evaluate(
            model, test_loader, eval_criterion, DEVICE, ds.tgt_mean, ds.tgt_std)
        out[name] = {t: {k: float(metrics[t][k]) for k in METRICS} for t in TASK_NAMES}
        out[name]['consistency'] = consistency_metrics(model, grids)
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    for name, fn in SK_MODELS:
        res, _ = fn(X_train, y_train_raw, X_test, y_test_raw, TASK_NAMES)
        out[name] = {t: {k: float(res[t][k]) for k in METRICS} for t in TASK_NAMES}

    return out


def paired(runs, a, b, task=None):
    d = np.array([r2_of(r, a, task) - r2_of(r, b, task) for r in runs])
    n = len(d)
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n) if sd > 0 else float('inf')
    t = d.mean() / se if se not in (0, float('inf')) else float('nan')
    return {'mean_diff': float(d.mean()), 'std_diff': float(sd),
            'per_seed': d.tolist(), 'n_wins': int((d > 0).sum()),
            't_stat': float(t)}


def main():
    tw = load_frozen_weights()
    print(f'Frozen task weights: {tw}')
    print(f'Device: {DEVICE}')
    print(f'Seeds: {SEEDS}\n')

    ds = ThinFilmDataset(DATA_PATH)

    runs = []
    for k, seed in enumerate(SEEDS, 1):
        t0 = time.time()
        res = run_one_seed(ds, seed, tw)
        res['seed'] = seed
        runs.append(res)

        avg = {m: r2_of(res, m) for m in res if m != 'seed'}
        order = sorted(avg, key=lambda m: -avg[m])
        print(f'[{k}/{len(SEEDS)}] seed={seed}  ({time.time()-t0:.0f}s)')
        for m in order:
            print('    ' + f'{m:20s} {avg[m]:.3f}   '
                  + ' '.join(f'{res[m][t]["R2"]:.3f}' for t in TASK_NAMES))
        print()

    models = [m for m in runs[0] if m != 'seed']

    # ---- per-task mean / std for every metric --------------------------------
    stats = {}
    for m in models:
        entry = {'avg_r2': {'mean': float(np.mean([r2_of(r, m) for r in runs])),
                            'std': float(np.std([r2_of(r, m) for r in runs], ddof=1))},
                 'per_task': {}}
        for t in TASK_NAMES:
            entry['per_task'][TASK_LABEL[t]] = {}
            for k in METRICS:
                vals = [r[m][t][k] for r in runs]
                entry['per_task'][TASK_LABEL[t]][k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)),
                    'values': vals}
        stats[m] = entry

    comparisons = {
        'MT-KAN(full) - RandomForest': paired(runs, 'MT-KAN (full)', 'Random Forest'),
        'MT-KAN(full) - XGBoost': paired(runs, 'MT-KAN (full)', 'XGBoost'),
        'MT-KAN(full) - MLP': paired(runs, 'MT-KAN (full)', 'MLP'),
        'MT-KAN(full) - MT-KAN(neither)': paired(runs, 'MT-KAN (full)', 'MT-KAN (neither)'),
        'MT-KAN(neither) - MLP': paired(runs, 'MT-KAN (neither)', 'MLP'),
        'MT-KAN(full) - kNN': paired(runs, 'MT-KAN (full)', 'kNN'),
        'MT-KAN(full) - SVR': paired(runs, 'MT-KAN (full)', 'SVR'),
        'MT-KAN(full) - ElasticNet': paired(runs, 'MT-KAN (full)', 'ElasticNet'),
    }
    paired_tasks = {}
    for base in ['MLP', 'Random Forest', 'XGBoost', 'MT-KAN (neither)']:
        paired_tasks[base] = {
            TASK_LABEL[t]: paired(runs, 'MT-KAN (full)', base, task=t)
            for t in TASK_NAMES}

    payload = {'seeds': SEEDS, 'task_weights': tw, 'runs': runs,
               'stats': stats, 'paired_comparisons': comparisons,
               'paired_per_task': paired_tasks}
    path = os.path.join(OUT_DIR, 'multiseed_results.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)

    # ---- console report ------------------------------------------------------
    print('=' * 96)
    print('MULTI-SEED MEAN +/- STD  (5 seeds, split + init both varied)')
    print('=' * 96)
    hdr = f'{"model":20s}' + ''.join(f'{TASK_LABEL[t]:>14s}' for t in TASK_NAMES) + f'{"Avg":>16s}'
    print(hdr); print('-' * len(hdr))
    for m in sorted(models, key=lambda x: -stats[x]['avg_r2']['mean']):
        row = f'{m:20s}'
        for t in TASK_NAMES:
            s = stats[m]['per_task'][TASK_LABEL[t]]['R2']
            row += f'{s["mean"]:>8.3f}+-{s["std"]:<5.3f}'
        a = stats[m]['avg_r2']
        row += f'{a["mean"]:>9.3f}+-{a["std"]:<5.3f}'
        print(row)

    print('\nMEAN RMSE (5 seeds)')
    hdr = f'{"model":20s}' + ''.join(f'{TASK_LABEL[t]:>14s}' for t in TASK_NAMES)
    print(hdr); print('-' * len(hdr))
    for m in sorted(models, key=lambda x: -stats[x]['avg_r2']['mean']):
        row = f'{m:20s}'
        for t in TASK_NAMES:
            s = stats[m]['per_task'][TASK_LABEL[t]]['RMSE']
            row += f'{s["mean"]:>8.3f}+-{s["std"]:<5.3f}'
        print(row)

    print('\n' + '=' * 96)
    print('PAIRED COMPARISONS (MT-KAN minus baseline, within each seed, avg R2)')
    print('=' * 96)
    print(f'{"comparison":36s}{"mean diff":>11s}{"std":>9s}{"wins":>7s}{"t":>8s}')
    print('-' * 96)
    for name, c in comparisons.items():
        print(f'{name:36s}{c["mean_diff"]:>11.4f}{c["std_diff"]:>9.4f}'
              f'{c["n_wins"]:>4d}/{len(SEEDS)}{c["t_stat"]:>8.2f}')

    print('\nPer-seed average R^2 (shows the split-to-split shift)')
    print('-' * 96)
    print(f'{"seed":>8s}' + ''.join(f'{m[:11]:>13s}' for m in models))
    for r in runs:
        line = f'{r["seed"]:>8d}'
        for m in models:
            line += f'{r2_of(r, m):>13.3f}'
        print(line)

    print('\nSaved ->', path)


if __name__ == '__main__':
    main()
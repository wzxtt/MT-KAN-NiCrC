"""
Stage 2: final training run. Produces every number reported in the paper.

Protocol
--------
The per-task loss weights were frozen by select_task_weights.py using the
validation split of the *old* (seed 42) partition. To make sure the reported
test score cannot be influenced by anything chosen during model development,
this script builds a FRESH partition with a different seed (SEED_FINAL) and
evaluates on its test split exactly once.

Reported here:
  * MT-KAN (physics prior + frozen task weights)
  * capacity-matched MLP (14,149 parameters, no physics prior)
  * RF / XGBoost / SVR / kNN / ElasticNet
  * 2x2 factorial ablation: physics prior on/off x task weights on/off
  * test metrics split by data source (exact / synthetic / digitised)
"""
import os
import sys
import json
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
TASK_LABEL = {'Sa_nm': 'Sa', 'H_GPa': 'H', 'E_GPa': 'E',
              'stress_GPa': 'sigma', 'H_Estar': 'H/E*'}
FEAT_NAMES = ['C2H2_flow', 'bias_voltage', 'dep_time', 'material']
SOURCES = ['exact', 'synthetic', 'digitized']

SEED_FINAL = 2026
N_EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 40


def load_frozen_weights():
    path = os.path.join(OUT_DIR, 'task_weights.json')
    with open(path) as f:
        payload = json.load(f)
    print(f"Frozen task weights from validation selection: "
          f"{payload['selected_name']} -> {payload['selected_weights']} "
          f"(val R^2={payload['selected_val_r2']:.4f})")
    return payload['selected_weights'], payload


def main():
    tw, tw_meta = load_frozen_weights()

    np.random.seed(SEED_FINAL)
    torch.manual_seed(SEED_FINAL)

    print('=' * 70)
    print(f'Stage 2 - FINAL run  (device={DEVICE}, fresh split seed={SEED_FINAL})')
    print('=' * 70)

    ds = ThinFilmDataset(DATA_PATH)
    phys_ctx = sample_phys_contexts(ds)
    all_X = ds.features
    all_Y = ds.targets * ds.tgt_std[None, :] + ds.tgt_mean[None, :]
    all_Y_norm = ds.targets
    all_mask = ds.valid_mask
    raw_denorm = np.where(all_mask, all_Y, np.nan)

    idx = np.arange(len(ds))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED_FINAL)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=SEED_FINAL)
    print(f'Train {len(train_idx)}  Val {len(val_idx)}  Test {len(test_idx)}')

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE)

    X_train, X_test = all_X[train_idx], all_X[test_idx]
    y_train_raw = raw_denorm[train_idx]
    y_test_raw = raw_denorm[test_idx]
    y_test = all_Y[test_idx]
    mask_test = all_mask[test_idx]

    # Record split composition for the honest data-provenance disclosure
    src_all = ds.df['source'].values
    composition = {}
    for sname, sidx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        sub = ds.df.iloc[sidx]
        composition[sname] = {
            'n_samples': int(len(sidx)),
            'by_source': {s: int((sub['source'] == s).sum()) for s in SOURCES},
            'valid_by_source_task': {
                s: {TASK_LABEL[t]: int(sub.loc[sub['source'] == s, t].notna().sum())
                    for t in TASK_NAMES}
                for s in SOURCES
            },
        }
    with open(os.path.join(OUT_DIR, 'split_composition.json'), 'w') as f:
        json.dump({'seed': SEED_FINAL, 'splits': composition}, f, indent=2)

    eval_criterion = WeightedMSELoss()

    # ---------------- MT-KAN (full: physics + frozen task weights) -------------
    print('\n[1/4] MT-KAN (physics prior + frozen task weights)')
    np.random.seed(SEED_FINAL); torch.manual_seed(SEED_FINAL)
    kan_model = MultiTaskKAN(n_input=4, n_hidden=64, n_tasks=5).to(DEVICE)
    n_params_kan = sum(p.numel() for p in kan_model.parameters())
    kan_hist, _, _, _, kan_best = train_model(
        kan_model, train_loader, val_loader, test_loader, DEVICE,
        ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS, model_name='MT-KAN',
        patience=PATIENCE, use_physics=True, task_weights=tw,
        feat_mean=ds.feat_mean, feat_std=ds.feat_std,
        phys_contexts=phys_ctx)
    _, kan_metrics, kan_preds, _ = evaluate(
        kan_model, test_loader, eval_criterion, DEVICE, ds.tgt_mean, ds.tgt_std)
    torch.save(kan_best, os.path.join(OUT_DIR, 'mt_kan_best.pth'))

    # ---------------- Capacity-matched MLP ------------------------------------
    print('\n[2/4] MLP baseline (capacity-matched)')
    np.random.seed(SEED_FINAL); torch.manual_seed(SEED_FINAL)
    mlp_model = MLPBaseline(n_input=4, n_tasks=5).to(DEVICE)
    n_params_mlp = sum(p.numel() for p in mlp_model.parameters())
    print(f'  MT-KAN {n_params_kan:,} params | MLP {n_params_mlp:,} params '
          f'(ratio {n_params_kan / n_params_mlp:.2f})')
    mlp_hist, _, _, _, _ = train_model(
        mlp_model, train_loader, val_loader, test_loader, DEVICE,
        ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS, model_name='MLP',
        patience=PATIENCE, use_physics=False)
    _, mlp_metrics, mlp_preds, _ = evaluate(
        mlp_model, test_loader, eval_criterion, DEVICE, ds.tgt_mean, ds.tgt_std)

    # ---------------- Classical baselines --------------------------------------
    print('\n[3/4] Classical baselines')
    sk_metrics, sk_preds = {}, {}
    for name, fn in [('Random Forest', train_random_forest_baseline),
                     ('XGBoost', train_xgboost_baseline),
                     ('SVR', train_svr_baseline),
                     ('kNN', train_knn_baseline),
                     ('ElasticNet', train_elasticnet_baseline)]:
        res, preds = fn(X_train, y_train_raw, X_test, y_test_raw, TASK_NAMES)
        sk_metrics[name] = {k: v for k, v in res.items() if k in TASK_NAMES}
        sk_preds[name] = preds
        avg = np.mean([m['R2'] for m in sk_metrics[name].values()])
        print(f'  {name:16s} avg R^2 = {avg:.4f}')

    # ---------------- 2x2 ablation --------------------------------------------
    print('\n[4/4] 2x2 ablation (physics prior x task weights)')
    ablation_cells = {
        'physics + weights': (True, tw),
        'physics only':      (True, None),
        'weights only':      (False, tw),
        'neither':           (False, None),
    }
    ablation_metrics, ablation_hist = {}, {}
    for cell, (use_phys, cell_tw) in ablation_cells.items():
        np.random.seed(SEED_FINAL); torch.manual_seed(SEED_FINAL)
        m = MultiTaskKAN(n_input=4, n_hidden=64, n_tasks=5).to(DEVICE)
        h, tm, _, _, _ = train_model(
            m, train_loader, val_loader, test_loader, DEVICE,
            ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS,
            model_name=f'MT-KAN[{cell}]', patience=PATIENCE,
            use_physics=use_phys, task_weights=cell_tw,
            feat_mean=ds.feat_mean, feat_std=ds.feat_std,
            phys_contexts=phys_ctx)
        ablation_metrics[cell] = tm
        ablation_hist[cell] = h
    ablation_metrics['MLP'] = mlp_metrics

    # ---------------- Assemble main table -------------------------------------
    all_metrics = {'MT-KAN': kan_metrics, 'MLP': mlp_metrics, **sk_metrics}
    all_preds_dict = {'MT-KAN': kan_preds, 'MLP': mlp_preds, **sk_preds}
    model_names = list(all_metrics.keys())

    results_summary = {
        'meta': {
            'seed': SEED_FINAL,
            'n_params_mt_kan': int(n_params_kan),
            'n_params_mlp': int(n_params_mlp),
            'task_weights': tw,
            'task_weight_selection': tw_meta['selection_criterion'],
        },
        'models': {n: {t: all_metrics[n][t] for t in TASK_NAMES if t in all_metrics[n]}
                   for n in model_names},
        'ablation': {c: {t: ablation_metrics[c][t] for t in TASK_NAMES
                         if t in ablation_metrics[c]}
                     for c in ablation_cells},
    }
    with open(os.path.join(OUT_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    # ---------------- Per-source test metrics --------------------------------
    src_test = src_all[test_idx]
    per_source = {}
    for mname in ['MT-KAN', 'MLP', 'Random Forest', 'XGBoost']:
        if mname not in all_preds_dict:
            continue
        per_source[mname] = {}
        P = all_preds_dict[mname]
        for s in SOURCES:
            smask = (src_test == s)
            entry = {'n_samples': int(smask.sum()), 'tasks': {}}
            for t in TASK_NAMES:
                sel = smask & mask_test[:, TASK_NAMES.index(t)]
                n = int(sel.sum())
                rec = {'n': n}
                if n >= 3:
                    yt, yp = y_test[sel, TASK_NAMES.index(t)], P[sel, TASK_NAMES.index(t)]
                    ok = ~np.isnan(yt) & ~np.isnan(yp)
                    if ok.sum() >= 3:
                        yt, yp = yt[ok], yp[ok]
                        ss_res = np.sum((yt - yp) ** 2)
                        ss_tot = np.sum((yt - yt.mean()) ** 2)
                        rec.update({'R2': float(1 - ss_res / max(ss_tot, 1e-12)),
                                    'RMSE': float(np.sqrt(np.mean((yt - yp) ** 2)))})
                entry['tasks'][TASK_LABEL[t]] = rec
            per_source[mname][s] = entry
    with open(os.path.join(OUT_DIR, 'per_source_metrics.json'), 'w') as f:
        json.dump(per_source, f, indent=2, default=str)

    # ---------------- Console summary ----------------------------------------
    print('\n' + '=' * 70)
    print('FINAL TEST-SET R^2')
    print('=' * 70)
    hdr = f'{"model":18s}' + ''.join(f'{TASK_LABEL[t]:>9s}' for t in TASK_NAMES) + f'{"Avg":>9s}'
    print(hdr); print('-' * len(hdr))
    for n in sorted(model_names, key=lambda k: -np.mean(
            [all_metrics[k][t]['R2'] for t in TASK_NAMES if t in all_metrics[k]])):
        vals = [all_metrics[n][t]['R2'] for t in TASK_NAMES]
        print(f'{n:18s}' + ''.join(f'{v:9.3f}' for v in vals) + f'{np.mean(vals):9.3f}')

    print('\n2x2 ABLATION (test R^2)')
    print('-' * 70)
    for c in list(ablation_cells) + ['MLP']:
        vals = [ablation_metrics[c][t]['R2'] for t in TASK_NAMES]
        print(f'{c:20s}' + ''.join(f'{v:9.3f}' for v in vals) + f'{np.mean(vals):9.3f}')

    print('\nPER-SOURCE TEST R^2 (MT-KAN)')
    print('-' * 70)
    for s in SOURCES:
        cells = []
        for t in TASK_NAMES:
            rec = per_source['MT-KAN'][s]['tasks'][TASK_LABEL[t]]
            cells.append(f"{rec.get('R2', float('nan')):6.2f}(n={rec['n']:3d})")
        print(f'{s:12s}' + ''.join(f'{c:>16s}' for c in cells))

    # ---------------- Figures -----------------------------------------------
    source_labels = {s: np.array([s_ == s for s_ in src_test]) for s in SOURCES}
    history_dict = {'MT-KAN': kan_hist, 'MLP': mlp_hist,
                    'physics + weights': ablation_hist['physics + weights'],
                    'weights only': ablation_hist['weights only']}
    generate_all_figures(
        all_metrics=all_metrics, all_preds_dict=all_preds_dict,
        all_targets=y_test, history_dict=history_dict, model_names=model_names,
        task_names=TASK_NAMES, feature_names=FEAT_NAMES, model=kan_model, ds=ds,
        source_labels=source_labels, ablation_metrics=ablation_metrics,
        save_dir=FIG_DIR, mask_test=mask_test, X_test=X_test)

    print('\n' + '=' * 70)
    print('Done. Results ->', OUT_DIR)
    print('=' * 70)


if __name__ == '__main__':
    main()
"""
Stage 3b: complete the 2x2 ablation across all five seeds.

multiseed.py already ran 'physics + weights' (full) and 'neither' per seed.
This runs the two missing cells - 'physics only' and 'weights only' - on the
same five partitions and merges everything into a single 2x2 summary, so that
Fig. 5 / Table 2 can report five-seed means for all four cells.
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
from kan_model import MultiTaskKAN
from train_eval import (train_model, evaluate, WeightedMSELoss,
                        sample_phys_contexts, build_consistency_grids,
                        consistency_metrics)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join(BASE_DIR, 'unified_dataset', 'training_dataset.csv')
OUT = os.path.join(BASE_DIR, 'experiments', 'results', 'multiseed_results.json')

TASK_NAMES = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
TASK_LABEL = {'Sa_nm': 'Sa', 'H_GPa': 'H', 'E_GPa': 'E',
              'stress_GPa': 'sigma', 'H_Estar': 'H/E*'}
METRICS = ('R2', 'RMSE', 'MAE')

SEEDS = [2026, 2027, 2028, 2029, 2030]
N_EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 40

CELLS = [('physics only', True, None), ('weights only', False, 'TW')]


def main():
    payload = json.load(open(OUT))
    tw = payload['task_weights']
    runs = payload['runs']
    seeds_in_runs = [r['seed'] for r in runs]
    assert seeds_in_runs == SEEDS, f'seed mismatch: {seeds_in_runs}'

    ds = ThinFilmDataset(DATA_PATH)
    phys_ctx = sample_phys_contexts(ds)
    grids = build_consistency_grids(ds)
    all_X = ds.features
    all_Y_norm = ds.targets
    all_Y = all_Y_norm * ds.tgt_std[None, :] + ds.tgt_mean[None, :]
    raw_denorm = np.where(ds.valid_mask, all_Y, np.nan)

    print(f'Seed run-order: {SEEDS}\n')
    for i, seed in enumerate(SEEDS):
        idx = np.arange(len(ds))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=seed)
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE)

        for cell, use_phys, cell_tw in CELLS:
            t0 = time.time()
            cell_tw_val = tw if cell_tw == 'TW' else cell_tw
            np.random.seed(seed); torch.manual_seed(seed)
            m = MultiTaskKAN(n_input=4, n_tasks=5).to(DEVICE)
            train_model(m, train_loader, val_loader, test_loader, DEVICE,
                        ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS,
                        model_name=f'{cell}[{seed}]', patience=PATIENCE,
                        use_physics=use_phys, task_weights=cell_tw_val,
                        feat_mean=ds.feat_mean, feat_std=ds.feat_std,
                        phys_contexts=phys_ctx)
            _, metrics, _, _ = evaluate(m, test_loader, WeightedMSELoss(), DEVICE,
                                        ds.tgt_mean, ds.tgt_std)
            rec = {t: {k: float(metrics[t][k]) for k in METRICS} for t in TASK_NAMES}
            rec['consistency'] = consistency_metrics(m, grids)
            runs[i][cell] = rec
            avg = np.mean([rec[t]['R2'] for t in TASK_NAMES])
            cm = rec['consistency']
            print(f'seed {seed} {cell:16s} avg R2 = {avg:.3f}  '
                  f'Sa_viol={cm["sa_viol"]*100:.1f}% H_viol={cm["h_viol"]*100:.1f}%  '
                  f'({time.time()-t0:.0f}s)')
            del m
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        print()

    # checkpoint the freshly trained cells before any aggregation work
    payload['runs'] = runs
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)

    # ---- aggregate all four cells + MLP -------------------------------------
    FULL = 'MT-KAN (full)'
    NEITHER = 'MT-KAN (neither)'
    CELL_KEY = {'physics + weights': FULL, 'physics only': 'physics only',
                'weights only': 'weights only', 'neither': NEITHER, 'MLP': 'MLP'}
    cells = ['physics + weights', 'physics only', 'weights only', 'neither', 'MLP']
    ablation = {}
    for c in cells:
        key = CELL_KEY[c]
        if not all(key in r for r in runs):
            continue
        entry = {'per_task': {}, 'avg_r2': {}}
        for t in TASK_NAMES:
            entry['per_task'][TASK_LABEL[t]] = {}
            for k in METRICS:
                vals = [r[key][t][k] for r in runs]
                entry['per_task'][TASK_LABEL[t]][k] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals, ddof=1)),
                    'values': vals}
        avgs = [np.mean([r[key][t]['R2'] for t in TASK_NAMES]) for r in runs]
        entry['avg_r2'] = {'mean': float(np.mean(avgs)),
                           'std': float(np.std(avgs, ddof=1))}
        ablation[c] = entry

    # paired: full vs each cell (physics effect, weights effect)
    def paired(a, b, task=None):
        keys = TASK_NAMES if task is None else [task]
        d = np.array([np.mean([r[a][t]['R2'] for t in keys])
                      - np.mean([r[b][t]['R2'] for t in keys]) for r in runs])
        sd = d.std(ddof=1)
        se = sd / np.sqrt(len(d))
        return {'mean_diff': float(d.mean()), 'std_diff': float(sd),
                'n_wins': int((d > 0).sum()),
                't_stat': float(d.mean() / se) if se > 0 else float('nan')}

    comparisons = {
        'full - neither': paired(FULL, NEITHER),
        'full - physics only': paired(FULL, 'physics only'),
        'full - weights only': paired(FULL, 'weights only'),
        'weights effect (weights only - neither)': paired('weights only', NEITHER),
        'physics effect (physics only - neither)': paired('physics only', NEITHER),
    }

    payload['runs'] = runs
    payload['ablation_2x2'] = ablation
    payload['ablation_paired'] = comparisons
    with open(OUT, 'w') as f:
        json.dump(payload, f, indent=2)

    print('=' * 84)
    print('2x2 ABLATION, 5-seed mean +/- std')
    print('=' * 84)
    hdr = f'{"cell":16s}' + ''.join(f'{TASK_LABEL[t]:>14s}' for t in TASK_NAMES) + f'{"Avg":>16s}'
    print(hdr); print('-' * len(hdr))
    for c in cells:
        if c not in ablation:
            continue
        row = f'{c:16s}'
        for t in TASK_NAMES:
            s = ablation[c]['per_task'][TASK_LABEL[t]]['R2']
            row += f'{s["mean"]:>8.3f}+-{s["std"]:<5.3f}'
        a = ablation[c]['avg_r2']
        row += f'{a["mean"]:>9.3f}+-{a["std"]:<5.3f}'
        print(row)

    print('\nPAIRED (n=5, df=4, t_crit 0.05 = 2.776)')
    print('-' * 84)
    for name, c in comparisons.items():
        print(f'{name:56s} {c["mean_diff"]:+.4f} +- {c["std_diff"]:.4f}  '
              f'wins {c["n_wins"]}/5  t={c["t_stat"]:+.2f}')

    print('\nSaved ->', OUT)


if __name__ == '__main__':
    main()
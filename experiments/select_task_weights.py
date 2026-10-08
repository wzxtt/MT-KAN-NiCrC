"""
Stage 1: select per-task loss weights on the VALIDATION set only.

Why this script exists
----------------------
The task weights [1, 2.5, 1, 2.5, 2.0] that produced the previous headline
numbers were chosen after inspecting test-set results. That is test-set
leakage: the reported test score is then no longer an unbiased estimate.

This script redoes the selection properly. Every candidate is trained on the
training split and scored by mean validation R^2. The test split is never
read here. The winning vector is written to results/task_weights.json and
frozen; the final numbers are then produced once, by final_run.py.

Usage:
    python select_task_weights.py
"""
import os
import sys
import json
import time
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

from data_loader import ThinFilmDataset
from kan_model import MultiTaskKAN
from train_eval import train_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join(BASE_DIR, 'unified_dataset', 'training_dataset.csv')
OUT_DIR = os.path.join(BASE_DIR, 'experiments', 'results')

TASK_NAMES = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
SEED = 42
N_EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 40

# Candidate set. Includes the uniform vector and the previously test-selected
# vector so we can report whether a validation-only protocol would have
# arrived at the same choice.
CANDIDATES = {
    'uniform':            [1.0, 1.0, 1.0, 1.0, 1.0],
    'mild':               [1.0, 1.5, 1.0, 1.5, 1.25],
    'moderate':           [1.0, 2.0, 1.0, 2.0, 1.5],
    'previous_test_pick': [1.0, 2.5, 1.0, 2.5, 2.0],
    'strong':             [1.0, 3.0, 1.0, 3.0, 2.5],
    'sa_heavy':           [2.0, 1.0, 1.0, 1.0, 1.0],
}


def main():
    ds = ThinFilmDataset(DATA_PATH)

    # Build the split with the same seed as the original work, but hand the
    # test indices to nobody: this stage never touches them.
    idx = np.arange(len(ds))
    train_idx, _held_out = train_test_split(idx, test_size=0.2, random_state=SEED)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=SEED)

    from torch.utils.data import Subset, DataLoader
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE)

    print('=' * 68)
    print(f'Stage 1 - task-weight selection on VALIDATION only  (device={DEVICE})')
    print(f'train={len(train_idx)}  val={len(val_idx)}  (test split withheld)')
    print('=' * 68)

    rows = []
    for name, tw in CANDIDATES.items():
        # Re-seed per candidate so differences reflect the weights, not init luck
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        model = MultiTaskKAN(n_input=4, n_hidden=64, n_tasks=5).to(DEVICE)
        t0 = time.time()
        history, _, _, _, _ = train_model(
            model, train_loader, val_loader, None, DEVICE,
            ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS,
            model_name=f'MT-KAN[{name}]', patience=PATIENCE,
            use_physics=True, task_weights=tw)

        best_val_r2 = history['best_val_r2']
        elapsed = time.time() - t0
        rows.append({'name': name, 'weights': tw, 'val_r2': float(best_val_r2),
                     'best_epoch': int(history['best_epoch']),
                     'seconds': round(elapsed, 1)})
        print(f'  {name:20s} w={tw}  best_val_R2={best_val_r2:.4f}  '
              f'(stopped @ep {history["best_epoch"]}, {elapsed:.0f}s)')

    rows.sort(key=lambda r: -r['val_r2'])
    winner = rows[0]
    print('\n' + '-' * 68)
    print('Ranking by validation R^2:')
    for r in rows:
        mark = '  <-- selected' if r is winner else ''
        print(f'  {r["val_r2"]:.4f}  {r["name"]:20s}{mark}')
    print('-' * 68)

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        'selection_criterion': 'mean validation R^2 over the five tasks',
        'seed': SEED,
        'candidates': rows,
        'selected_name': winner['name'],
        'selected_weights': winner['weights'],
        'selected_val_r2': winner['val_r2'],
        'note': ('Chosen on the validation split only. The test split was not '
                 'used for any selection decision in this stage.'),
    }
    path = os.path.join(OUT_DIR, 'task_weights.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'\nSelected: {winner["name"]} -> {winner["weights"]}')
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
"""
Pilot: sweep lambda for the sweep-based physics penalty (v2).

Trains the headline cell (physics + task weights, seed 2026) for several
lambda values and reports accuracy vs physical-consistency metrics.

Reference values (old loss / no physics, seed 2026):
  full (old loss):   R2 = 0.645, Sa_viol = 45.8%, H_viol = 12.5%
  neither:           R2 = 0.676, Sa_viol = 10.2%, H_viol = 5.2%
Published full-model 5-seed mean: R2 = 0.671 +- 0.028.

Goal: Sa_viol / H_viol clearly below 'neither' with |dR2| <= ~0.01.
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
TW_PATH = os.path.join(BASE_DIR, 'experiments', 'results', 'task_weights.json')

TASK_NAMES = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
SEEDS = [int(s) for s in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['2026'])]
LAMBDAS = [float(s) for s in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['0.05', '0.2', '1.0'])]
N_EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 40


def main():
    tw = json.load(open(TW_PATH))['selected_weights']
    ds = ThinFilmDataset(DATA_PATH)
    ctx = sample_phys_contexts(ds)
    grids = build_consistency_grids(ds)

    for seed in SEEDS:
        idx = np.arange(len(ds))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=seed)
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE)

        for lam in LAMBDAS:
            t0 = time.time()
            np.random.seed(seed)
            torch.manual_seed(seed)
            m = MultiTaskKAN(n_input=4, n_tasks=5).to(DEVICE)
            hist, test_metrics, _, _, _ = train_model(
                m, train_loader, val_loader, test_loader, DEVICE,
                ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS,
                model_name=f'full lam={lam} seed={seed}', patience=PATIENCE,
                use_physics=True, task_weights=tw, lambda_physics=lam,
                feat_mean=ds.feat_mean, feat_std=ds.feat_std,
                phys_contexts=ctx)
            avg_r2 = float(np.mean([test_metrics[t]['R2'] for t in TASK_NAMES]))
            cm = consistency_metrics(m, grids)
            print(f'seed={seed} lam={lam:<5g} R2={avg_r2:.3f}  '
                  f'Sa_viol={cm["sa_viol"]*100:5.1f}% ({cm["sa_viol_m"]*100:4.1f}%>m)  '
                  f'H_viol={cm["h_viol"]*100:5.1f}% ({cm["h_viol_m"]*100:4.1f}%>m)  '
                  f'Sa_pen={cm["sa_pen"]:.4f} H_pen={cm["h_pen"]:.4f}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
            del m
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        print(flush=True)


if __name__ == '__main__':
    main()

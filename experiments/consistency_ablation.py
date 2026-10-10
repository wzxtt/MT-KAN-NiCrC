"""
Consistency evaluation of the 2x2 ablation cells.

The accuracy ablation (Table 2) shows the physics term has no measurable
R^2 effect. This script measures what the term is actually designed to do:
physical consistency of the predicted response surfaces.

Two metrics, evaluated on a fixed sweep grid (identical for every cell):

  1. Sa-time monotonicity violation rate (%)
     Predicted roughness must decrease with deposition time.
     Sweep time over the NiCrC range at fixed context; a violation is any
     adjacent step where predicted Sa increases. This is exactly the
     constraint penalised by the physics loss (relu(diff) on time-sorted
     predictions).

  2. H-bias extremum violation rate (%)
     Hardness must be lower at extreme bias than at the optimal bias
     (Gaussian-like peak). Sweep bias over the NiCrC range; a violation is
     any extreme point (12 most extreme |bias| values) where predicted H
     exceeds predicted H at the bias centre (normalised 0, the centre used
     by the physics loss).

Contexts: 32 NiCrC rows (with real time data) sampled once with a fixed RNG,
so every cell/seed is scored on exactly the same inputs.

Retrains all four cells for all five seeds (same hyper-parameters and
splits as ablation_multiseed.py) because no per-cell checkpoints were kept.
Results go to experiments/results/consistency_results.json.
"""
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.join(BASE_DIR, 'experiments'))

from data_loader import ThinFilmDataset
from kan_model import MultiTaskKAN
from train_eval import (train_model, evaluate, WeightedMSELoss,
                        sample_phys_contexts)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = os.path.join(BASE_DIR, 'unified_dataset', 'training_dataset.csv')
SRC = os.path.join(BASE_DIR, 'experiments', 'results', 'multiseed_results.json')
OUT = os.path.join(BASE_DIR, 'experiments', 'results', 'consistency_results.json')

SEEDS = [2026, 2027, 2028, 2029, 2030]
N_EPOCHS = 500
BATCH_SIZE = 64
PATIENCE = 40
N_CTX = 32          # sweep contexts
N_TIME = 60         # time sweep points
N_BIAS = 60         # bias sweep points
N_EXTREME = 12      # most-extreme bias points checked for the H peak
CONTEXT_SEED = 0    # fixed context sampling

CELLS = [('physics + weights', True, 'TW'),
         ('physics only', True, None),
         ('weights only', False, 'TW'),
         ('neither', False, None)]


def build_grids(ds, df):
    """Sweep grids + fixed contexts, in normalised feature space."""
    ni = df[df['material'] == 'NiCrC'].copy()
    time_col = ni['time_s']
    ni = ni[time_col.notna()]
    rs = np.random.RandomState(CONTEXT_SEED)
    ctx_pos = rs.choice(len(ni), size=N_CTX, replace=False)

    flow_mean, bias_mean, time_mean = ds.feat_mean[:3]
    flow_std, bias_std, time_std = ds.feat_std[:3]
    rows = ni.iloc[ctx_pos]
    ctx_flow = ((rows['C2H2_sccm'].values.astype(np.float64) - flow_mean) / flow_std)
    ctx_bias = ((rows['bias_V'].values.astype(np.float64) - bias_mean) / bias_std)
    ctx_time = ((rows['time_s'].values.astype(np.float64) - time_mean) / time_std)

    time_raw = np.linspace(10.0, 3600.0, N_TIME)
    time_norm = (time_raw - time_mean) / time_std
    bias_raw = np.linspace(0.0, 800.0, N_BIAS)
    bias_norm = (bias_raw - bias_mean) / bias_std
    center_norm = 0.0            # physics-loss centre: normalised bias = 0

    # ---- time sweep grid: (N_CTX * N_TIME, 4)
    T = np.zeros((N_CTX * N_TIME, 4), dtype=np.float32)
    T[:, 0] = np.repeat(ctx_flow, N_TIME)
    T[:, 1] = np.repeat(ctx_bias, N_TIME)
    T[:, 2] = np.tile(time_norm, N_CTX)
    T[:, 3] = 0.0
    # ---- bias sweep grid: (N_CTX * N_BIAS, 4)
    B = np.zeros((N_CTX * N_BIAS, 4), dtype=np.float32)
    B[:, 0] = np.repeat(ctx_flow, N_BIAS)
    B[:, 1] = np.tile(bias_norm, N_CTX)
    B[:, 2] = np.repeat(ctx_time, N_BIAS)
    B[:, 3] = 0.0
    # ---- centre probes: (N_CTX, 4)
    C = np.zeros((N_CTX, 4), dtype=np.float32)
    C[:, 0] = ctx_flow
    C[:, 1] = center_norm
    C[:, 2] = ctx_time
    C[:, 3] = 0.0

    extreme_idx = np.argsort(-np.abs(bias_norm))[:N_EXTREME]   # into bias axis
    extreme_idx = np.sort(extreme_idx)
    return T, B, C, extreme_idx, time_raw, bias_raw, center_norm * bias_std + bias_mean


@torch.no_grad()
def consistency(model, T, B, C, extreme_idx):
    model.eval()
    def pred(arr):
        x = torch.from_numpy(arr).to(DEVICE)
        return model(x).cpu().numpy()

    # metric 1: Sa-time monotonicity (task 0)
    sa = pred(T).reshape(N_CTX, N_TIME, -1)[:, :, 0]
    diff = sa[:, 1:] - sa[:, :-1]
    sa_viol = float((diff > 0).mean())
    sa_pen = float(np.maximum(diff, 0).mean())

    # metric 2: H-bias extremum (task 1)
    h = pred(B).reshape(N_CTX, N_BIAS, -1)[:, :, 1]
    h_c = pred(C)[:, 1]                                # (N_CTX,)
    h_ext = h[:, extreme_idx]                          # (N_CTX, N_EXTREME)
    h_viol = float((h_ext > h_c[:, None]).mean())
    return {'sa_viol': sa_viol, 'sa_pen': sa_pen, 'h_viol': h_viol}


def main():
    payload = json.load(open(SRC))
    tw = payload['task_weights']
    df = pd.read_csv(DATA_PATH)

    ds = ThinFilmDataset(DATA_PATH)
    phys_ctx = sample_phys_contexts(ds)
    T, B, C, ext_idx, time_raw, bias_raw, center_raw = build_grids(ds, df)
    print(f'grids: time {time_raw[0]:.0f}-{time_raw[-1]:.0f}s, '
          f'bias {bias_raw[0]:.0f}-{bias_raw[-1]:.0f}V '
          f'(centre {center_raw:.0f}V), {len(ext_idx)} extreme bias points\n')

    runs = []
    for seed in SEEDS:
        idx = np.arange(len(ds))
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=seed)
        train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(ds, val_idx), batch_size=BATCH_SIZE)
        test_loader = DataLoader(Subset(ds, test_idx), batch_size=BATCH_SIZE)

        rec = {'seed': seed}
        for cell, use_phys, cell_tw in CELLS:
            t0 = time.time()
            cell_tw_val = tw if cell_tw == 'TW' else None
            np.random.seed(seed)
            torch.manual_seed(seed)
            m = MultiTaskKAN(n_input=4, n_tasks=5).to(DEVICE)
            hist, test_metrics, _, _, _ = train_model(
                m, train_loader, val_loader, test_loader, DEVICE,
                ds.tgt_mean, ds.tgt_std, n_epochs=N_EPOCHS,
                model_name=f'{cell}[{seed}]', patience=PATIENCE,
                use_physics=use_phys, task_weights=cell_tw_val,
                feat_mean=ds.feat_mean, feat_std=ds.feat_std,
                phys_contexts=phys_ctx)
            cm = consistency(m, T, B, C, ext_idx)
            cm['avg_r2'] = float(np.mean([v['R2'] for v in test_metrics.values()]))
            rec[cell] = cm
            print(f'seed {seed} {cell:18s} Sa_viol={cm["sa_viol"]*100:5.1f}%  '
                  f'H_viol={cm["h_viol"]*100:5.1f}%  R2={cm["avg_r2"]:.3f}  '
                  f'({time.time()-t0:.0f}s)', flush=True)
            del m
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
        runs.append(rec)
        print(flush=True)
        # checkpoint incrementally
        json.dump({'seeds': SEEDS, 'runs': runs}, open(OUT, 'w'), indent=2)

    # ---- paired statistics (n=5, df=4, t_crit=2.776) ----
    def paired(a_get, b_get, key):
        d = np.array([a_get(r)[key] - b_get(r)[key] for r in runs])
        sd = d.std(ddof=1)
        se = sd / np.sqrt(len(d))
        return {'mean_diff': float(d.mean()), 'std_diff': float(sd),
                't_stat': float(d.mean() / se) if se > 0 else None,
                'per_seed': [float(x) for x in d]}

    get = lambda cell: (lambda r: r[cell])
    stats = {
        'physics effect (physics only - neither)': {
            k: paired(get('physics only'), get('neither'), k) for k in ('sa_viol', 'h_viol')},
        'full - neither': {
            k: paired(get('physics + weights'), get('neither'), k) for k in ('sa_viol', 'h_viol')},
        'weights effect (weights only - neither)': {
            k: paired(get('weights only'), get('neither'), k) for k in ('sa_viol', 'h_viol')},
    }
    # cell means
    cell_means = {}
    for cell, _, _ in CELLS:
        cell_means[cell] = {k: {'mean': float(np.mean([r[cell][k] for r in runs])),
                                'std': float(np.std([r[cell][k] for r in runs], ddof=1))}
                            for k in ('sa_viol', 'h_viol', 'avg_r2')}

    out = {'seeds': SEEDS, 'n_ctx': N_CTX, 'n_time': N_TIME, 'n_bias': N_BIAS,
           'n_extreme': N_EXTREME, 'runs': runs,
           'cell_means': cell_means, 'paired': stats}
    json.dump(out, open(OUT, 'w'), indent=2)

    print('=' * 78)
    print('CONSISTENCY METRICS, 5-seed mean +/- std  (lower = more physical)')
    print('=' * 78)
    for cell, _, _ in CELLS:
        m = cell_means[cell]
        print(f'{cell:18s}  Sa-viol {m["sa_viol"]["mean"]*100:5.1f}+-{m["sa_viol"]["std"]*100:4.1f}%   '
              f'H-viol {m["h_viol"]["mean"]*100:5.1f}+-{m["h_viol"]["std"]*100:4.1f}%   '
              f'R2 {m["avg_r2"]["mean"]:.3f}')
    print('-' * 78)
    for name, ks in stats.items():
        for k, v in ks.items():
            print(f'{name:44s} {k:8s} diff={v["mean_diff"]*100:+6.2f}+-{v["std_diff"]*100:5.2f}pp  '
                  f't={v["t_stat"]:+.2f}' if v['t_stat'] is not None else
                  f'{name:44s} {k:8s} diff={v["mean_diff"]*100:+6.2f}pp (constant)')
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()

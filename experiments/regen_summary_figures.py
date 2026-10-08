"""
Regenerate the summary figures from FIVE-SEED means (fig4, fig5, fig11) and
rebuild fig8 so that it shows something computable from this dataset:

  fig4 - model comparison bars   : 5-seed mean R2 per task + average
  fig5 - 2x2 ablation bars       : 5-seed mean R2 for all four cells
  fig11 - RMSE/MAE heatmap       : 5-seed mean RMSE/MAE
  fig8 - LEFT  : pairwise co-observation counts (full dataset, ground truth)
         RIGHT : MT-KAN predicted correlation on the 918 test samples (seed 2026)

The old fig8 drew correlations of mean-imputed targets, which are ~0 by
construction and contradicted the text; it is replaced here.
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

BASE = r'C:\Users\28100\Desktop\双因素影响'
sys.path.insert(0, os.path.join(BASE, 'experiments'))

import figures
from data_loader import ThinFilmDataset
from kan_model import MultiTaskKAN

FIG_DIR = os.path.join(BASE, 'experiments', 'figures')
OUT_JSON = os.path.join(BASE, 'experiments', 'results', 'multiseed_results.json')
DATA = os.path.join(BASE, 'unified_dataset', 'training_dataset.csv')
CKPT = os.path.join(BASE, 'experiments', 'results', 'mt_kan_best.pth')

TASK_NAMES = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
TASK_LABEL = {'Sa_nm': 'Sa', 'H_GPa': 'H', 'E_GPa': 'E',
              'stress_GPa': 'sigma', 'H_Estar': 'H/E*'}
MODEL_MAP = {'MT-KAN (full)': 'MT-KAN', 'MT-KAN (neither)': None, 'MLP': 'MLP',
             'Random Forest': 'Random Forest', 'XGBoost': 'XGBoost',
             'kNN': 'kNN', 'SVR': 'SVR', 'ElasticNet': 'ElasticNet'}

figures.set_save_dir(FIG_DIR)


def main():
    payload = json.load(open(OUT_JSON))
    stats = payload['stats']
    abl = payload['ablation_2x2']

    # ---- fig4 / fig11 inputs: 5-seed means, 7 headline models ---------------
    all_metrics = {}
    for src, dst in MODEL_MAP.items():
        if dst is None:
            continue
        all_metrics[dst] = {
            t: {'R2': stats[src]['per_task'][TASK_LABEL[t]]['R2']['mean'],
                'RMSE': stats[src]['per_task'][TASK_LABEL[t]]['RMSE']['mean'],
                'MAE': stats[src]['per_task'][TASK_LABEL[t]]['MAE']['mean']}
            for t in TASK_NAMES}
    model_names = sorted(all_metrics,
                         key=lambda m: -np.mean([all_metrics[m][t]['R2'] for t in TASK_NAMES]))
    print('fig4/11 model order:', model_names)

    figures.fig_model_comparison(all_metrics, model_names, TASK_NAMES)
    figures.fig_rmse_mae_heatmap(all_metrics, model_names, TASK_NAMES)

    # ---- fig5 inputs: 4 ablation cells --------------------------------------
    ablation_metrics = {}
    for cell in figures.ABLATION_VARIANTS:
        if cell in abl:
            ablation_metrics[cell] = {
                t: {'R2': abl[cell]['per_task'][TASK_LABEL[t]]['R2']['mean']}
                for t in TASK_NAMES}
    figures.fig_ablation_study(ablation_metrics, TASK_NAMES)

    # ---- fig8: co-observation counts (left) + predicted correlation (right) --
    ds = ThinFilmDataset(DATA)
    raw = pd.read_csv(DATA)
    B = raw[TASK_NAMES].notna().astype(int).values          # (4587, 5)
    counts = B.T @ B                                        # co-observation

    idx = np.arange(len(ds))
    _, test_idx = train_test_split(idx, test_size=0.2, random_state=2026)
    X_test = torch.FloatTensor(ds.features[test_idx])
    model = MultiTaskKAN(n_input=4, n_tasks=5)
    best = torch.load(CKPT, map_location='cpu')
    if isinstance(best, dict) and 'state_dict' in best:
        best = best['state_dict']
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()

    corr_pred = np.corrcoef(preds.T)
    print('\npredicted correlation (test, n=%d):' % len(test_idx))
    names = ['Sa', 'H', 'E', 'sig', 'H/E*']
    print('     ' + ''.join(f'{n:>8s}' for n in names))
    for i, n in enumerate(names):
        print(f'{n:>4s} ' + ''.join(f'{corr_pred[i, j]:>8.2f}' for j in range(5)))

    off = counts[~np.eye(5, dtype=bool)]
    vmax = max(off.max(), 1)

    task_short = ['$S_a$', '$H$', '$E$', r'$\sigma$', '$H/E^*$']
    fig, axes = plt.subplots(1, 2, figsize=(figures.NATURE_DOUBLE[0],
                                            figures.NATURE_DOUBLE[0] / 2.0))
    sns.heatmap(counts, ax=axes[0], cmap='Blues', annot=counts, fmt='d',
                annot_kws={'fontsize': 6.5}, xticklabels=task_short,
                yticklabels=task_short, vmin=0, vmax=vmax, linewidths=0.5,
                linecolor='white', cbar_kws={'shrink': 0.8, 'label': 'n (capped at off-diagonal max)'})
    axes[0].set_title('Ground-truth co-observation (full dataset)', fontsize=10.5)
    sns.heatmap(corr_pred, ax=axes[1], cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                annot=True, fmt='.2f', annot_kws={'fontsize': 6.5},
                xticklabels=task_short, yticklabels=task_short, linewidths=0.5,
                linecolor='white', cbar_kws={'shrink': 0.8})
    axes[1].set_title('MT-KAN predicted correlation (test, n=918)', fontsize=10.5)
    plt.tight_layout()
    figures.save_figure(fig, 'fig8_correlation_matrix')

    print('\nDone ->', FIG_DIR)


if __name__ == '__main__':
    main()
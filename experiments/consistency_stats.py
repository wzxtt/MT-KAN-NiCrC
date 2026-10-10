"""
Consistency + accuracy summary across the four 2x2 ablation cells.

Reads experiments/results/multiseed_results.json (must already contain the
'consistency' key in each KAN/MLP run entry, i.e. produced by the patched
multiseed.py + ablation_multiseed.py) and prints:

  * per-cell 5-seed mean +- std for accuracy and the six consistency metrics
  * paired comparisons (n=5, df=4, t_crit=2.776) for the physics effect,
    the full-vs-neither contrast, and the weights effect

Also writes experiments/results/consistency_stats.json.
"""
import os
import json
import numpy as np

BASE = r'C:\Users\28100\Desktop\双因素影响'
SRC = os.path.join(BASE, 'experiments', 'results', 'multiseed_results.json')
OUT = os.path.join(BASE, 'experiments', 'results', 'consistency_stats.json')

CELLS = ['MT-KAN (full)', 'physics only', 'weights only', 'MT-KAN (neither)', 'MLP']
CELL_LABEL = {'MT-KAN (full)': 'physics + weights', 'physics only': 'physics only',
              'weights only': 'weights only', 'MT-KAN (neither)': 'neither',
              'MLP': 'MLP'}
TASKS = ['Sa_nm', 'H_GPa', 'E_GPa', 'stress_GPa', 'H_Estar']
CM_KEYS = ['sa_viol', 'sa_viol_m', 'sa_pen', 'h_viol', 'h_viol_m', 'h_pen']


def cell_of(runs, cell):
    return [r[cell] for r in runs if cell in r]


def paired(runs, a, b, key):
    """key: 'avg_r2' or a consistency metric inside entry['consistency']."""
    A, B = cell_of(runs, a), cell_of(runs, b)
    assert len(A) == 5 and len(B) == 5, (a, len(A), b, len(B))
    def get(e, k):
        if k == 'avg_r2':
            return float(np.mean([e[t]['R2'] for t in TASKS]))
        return float(e['consistency'][k])
    d = np.array([get(x, key) - get(y, key) for x, y in zip(A, B)])
    sd = d.std(ddof=1)
    se = sd / np.sqrt(len(d))
    return {'mean_diff': float(d.mean()), 'std_diff': float(sd),
            't_stat': float(d.mean() / se) if se > 0 else None,
            'wins': int((d < 0).sum()) if key != 'avg_r2' else int((d > 0).sum()),
            'per_seed': [float(x) for x in d]}


def main():
    payload = json.load(open(SRC))
    runs = payload['runs']

    cell_stats = {}
    for cell in CELLS:
        entries = cell_of(runs, cell)
        if not entries:
            print(f'skip (missing): {cell}')
            continue
        e = {}
        avgs = [float(np.mean([x[t]['R2'] for t in TASKS])) for x in entries]
        e['avg_r2'] = {'mean': float(np.mean(avgs)),
                       'std': float(np.std(avgs, ddof=1))}
        if 'consistency' in entries[0]:
            for k in CM_KEYS:
                vals = [float(x['consistency'][k]) for x in entries]
                e[k] = {'mean': float(np.mean(vals)),
                        'std': float(np.std(vals, ddof=1)), 'values': vals}
        cell_stats[cell] = e

    comparisons = {}
    for name, (a, b) in {
        'physics effect (physics only - neither)': ('physics only', 'MT-KAN (neither)'),
        'full - neither': ('MT-KAN (full)', 'MT-KAN (neither)'),
        'weights effect (weights only - neither)': ('weights only', 'MT-KAN (neither)'),
    }.items():
        comparisons[name] = {k: paired(runs, a, b, k)
                             for k in ['avg_r2'] + CM_KEYS}

    out = {'cell_stats': cell_stats, 'paired': comparisons}
    json.dump(out, open(OUT, 'w'), indent=2)

    print('=' * 88)
    print('5-seed mean +- std  (consistency: lower = more physical)')
    print('=' * 88)
    hdr = f'{"cell":20s}{"avg R2":>14s}{"Sa viol%":>14s}{"Sa viol>m%":>14s}{"H viol%":>14s}{"H viol>m%":>14s}'
    print(hdr)
    print('-' * 88)
    for cell in CELLS:
        if cell not in cell_stats:
            continue
        e = cell_stats[cell]
        def fm(k, pct=False):
            if k not in e:
                return f'{"-":>14s}'
            m, s = e[k]['mean'], e[k]['std']
            if pct:
                return f'{m*100:6.1f}+-{s*100:<5.1f}'
            return f'{m:6.3f}+-{s:<5.3f}'
        print(f'{CELL_LABEL[cell]:20s}{fm("avg_r2"):>14s}'
              f'{fm("sa_viol", True):>14s}{fm("sa_viol_m", True):>14s}'
              f'{fm("h_viol", True):>14s}{fm("h_viol_m", True):>14s}')

    print('-' * 88)
    print('PAIRED (n=5, df=4, |t| > 2.776 = significant at 0.05)')
    print('-' * 88)
    for name, ks in comparisons.items():
        print(name)
        for k, v in ks.items():
            if v['t_stat'] is None:
                print(f'    {k:12s} diff={v["mean_diff"]:+.4f} (constant)')
            else:
                unit = 'pp' if k in ('sa_viol', 'sa_viol_m', 'h_viol', 'h_viol_m') else ''
                scale = 100 if unit == 'pp' else 1
                print(f'    {k:12s} diff={v["mean_diff"]*scale:+.3f}{unit}'
                      f' +- {v["std_diff"]*scale:.3f}  t={v["t_stat"]:+.2f}'
                      f'  wins={v["wins"]}/5')
    print(f'\nsaved -> {OUT}')


if __name__ == '__main__':
    main()

# MT-KAN NiCrC Thin-Film Property Prediction: Data, Code, and Model

This deposit contains the dataset, analysis and training code, trained model
checkpoint, and numerical results underlying the manuscript:

> "Multi-task Kolmogorov-Arnold network for simultaneous prediction of
> multiple mechanical properties of FCVAD-deposited NiCrC thin films"

Authors: Bo Zhang, Xue Peng, Xuan Wu, Ke Zhuang, Tianchi Jiang.

## Contents

```
unified_dataset/training_dataset.csv   4587-sample dataset (data file)
experiments/*.py                       training, baseline, ablation, figure code
experiments/results/*.json             all reported metrics (5-seed results, ablation, summaries)
experiments/results/mt_kan_best.pth    trained MT-KAN checkpoint (PyTorch state_dict)
requirements.txt                       Python dependencies
LICENSE                                CC BY 4.0
```

## Data description

`training_dataset.csv` (4587 rows) combines three sources, flagged in the
`source` column:

| source      | rows | description                                      |
|-------------|------|--------------------------------------------------|
| exact       |   31 | experimental anchor points measured for this study |
| synthetic   | 2550 | systematically generated over the sampled parameter ranges |
| digitized   | 2006 | digitised literature records                      |

Columns:

- `material`: NiCrC or HEA (encoded as 0/1 by the loader)
- `series`: process series label (flow / bias / time)
- `C2H2_sccm`: acetylene flow rate (sccm)
- `bias_V`: substrate bias voltage (V)
- `time_s`: deposition time (s)
- `Sa_nm`: surface roughness Sa (nm)
- `H_GPa`: hardness (GPa)
- `E_GPa`: Young's modulus (GPa)
- `stress_GPa`: residual stress (GPa)
- `H_Estar`: wear-resistance index H/E* (dimensionless)
- `source`: exact / synthetic / digitized

Missing values are empty cells. Missing features are mean-imputed by the
loader; missing targets are masked out of the loss (never imputed).
Property coverage: Sa 1300, H 2087, E 310, stress 444, H/E* 392 observations.

## Reproducing the results

1. Install dependencies: `pip install -r requirements.txt`
2. Five-seed main experiment (Table 2 / Fig. 4-8 in the manuscript):
   `python experiments/multiseed.py`
   Writes `experiments/results/multiseed_results.json`.
3. 2x2 factorial ablation (Table 3):
   `python experiments/ablation_multiseed.py`
4. Figures: `python experiments/figures.py` and
   `python experiments/regen_summary_figures.py`.

All metrics are reported as mean +/- standard deviation over five seeds
(2026-2030), where each seed changes both the train/validation/test split and
the network initialisation; model comparisons are paired by seed.

## Environment

Python 3.10+ tested; key packages: PyTorch 2.x, numpy, pandas,
scikit-learn, xgboost, matplotlib, seaborn, tqdm.

## Licence

Released under the Creative Commons Attribution 4.0 International (CC BY 4.0)
licence. The `digitized` rows originate from published literature figures and
are redistributed for research use with attribution.

## Contact

Corresponding author: jtcedu@163.com

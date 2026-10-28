# MT-KAN NiCrC Thin-Film Property Prediction: Data, Code, and Model

This deposit contains the dataset, analysis and training code, trained model
checkpoint, and numerical results underlying the manuscript:

> "Multi-task Kolmogorov-Arnold network for simultaneous prediction of
> multiple mechanical properties of FCVAD-deposited NiCrC thin films"

Authors: Bo Zhang, Peng Zhang, Jieming Du, Ke Zhuang, Xiwei Ke, Junfu Lü.

## Contents

```
unified_dataset/training_dataset.csv   4587-sample dataset used in all experiments
dataset_build/                         provenance of the dataset: digitised source tables
                                       and the script that assembles them
experiments/*.py                       training, baseline, figure and analysis code
experiments/results/*.json             all reported metrics (five-seed results,
                                       per-source metrics, summary statistics)
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

## Reproducing the reported results

1. Install dependencies: `pip install -r requirements.txt`
2. Five-seed main experiment (Table 1 of the manuscript):
   `python experiments/multiseed.py`
   Writes `experiments/results/multiseed_results.json`.
3. Final model, trained checkpoint and summary metrics (Table 1, Figs. 5-8):
   `python experiments/final_run.py`
   Writes `results_summary.json`, `per_source_metrics.json`,
   `split_composition.json` and `mt_kan_best.pth` under `experiments/results/`.
4. Figures: `python experiments/figures.py` draws all panels; the five-seed
   summary panels (Fig. 5 and the correlation matrix) are produced by
   `python experiments/regen_summary_figures.py`.
5. Dataset provenance (optional): `python dataset_build/build_unified_dataset.py`
   regenerates the per-series tables and `combined_master.csv` from the
   digitised source tables shipped in `dataset_build/thesis_extracted_data/`.

All metrics are reported as mean +/- standard deviation over five seeds
(2026-2030), where each seed changes both the train/validation/test split and
the network initialisation; model comparisons are paired by seed.

### Figure files and manuscript numbering

The figure files keep their internal identifiers, which differ from the
numbered figures in the manuscript:

| Manuscript | File |
|---|---|
| Fig. 1 | `fig8_correlation_matrix` |
| Fig. 2 | `fig13_tsne` |
| Fig. 3 | `fig15_violin_source` |
| Fig. 4 | `fig7_spline_functions` |
| Fig. 5 | `fig11_rmse_mae_heatmap` |
| Fig. 6 | `fig1_parity_plots` |
| Fig. 7 | `fig6_parameter_maps` |
| Fig. 8 | `fig10_feature_importance` |

## Hyper-parameter selection and auxiliary analyses

- `experiments/select_task_weights.py`: selects the per-task loss weights used
  in training (`results/task_weights.json`).
- `experiments/pilot_physics.py`: pilot sweep used to select the physics-loss
  weight (lambda = 0.05).
- Exploratory scripts retained for completeness (`ablation_multiseed.py`,
  `consistency_ablation.py`, `consistency_stats.py`) produce the additional
  factor and consistency diagnostics stored in `results/`; these are not part
  of the numbers reported in the manuscript.

## Environment

Python 3.10+ tested; key packages: PyTorch 2.x, numpy, pandas,
scikit-learn, xgboost, matplotlib, seaborn, tqdm.

## Licence

Released under the Creative Commons Attribution 4.0 International (CC BY 4.0)
licence. The `digitized` rows originate from published literature figures and
are redistributed for research use with attribution.

## Contact

Corresponding author: Bo Zhang (zhang-bo@mail.tsinghua.edu.cn)

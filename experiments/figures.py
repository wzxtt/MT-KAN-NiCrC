"""
Publication-quality figures for MT-KAN paper.
7 models, Nature style, colorblind-friendly.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os, warnings, torch
warnings.filterwarnings('ignore')

# Nature journal style
NATURE_SINGLE = (89 / 25.4, 60 / 25.4)
NATURE_DOUBLE = (183 / 25.4, 120 / 25.4)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 300, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05, 'lines.linewidth': 0.75, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'xtick.minor.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# 7 distinct colorblind-friendly colors (IBM Carbon / Nature compatible palette)
C7 = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#999999']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']

_SAVE_DIR = r'C:\Users\28100\AppData\Local\Temp\opencode\figures_out'
# Actual final directory for the paper
_FINAL_DIR = r'C:\Users\28100\Desktop\双因素影响\experiments\figures'

def set_save_dir(path):
    global _FINAL_DIR
    _FINAL_DIR = path
    os.makedirs(_FINAL_DIR, exist_ok=True)

def save_figure(fig, name):
    # Save to ASCII-only temp path first, then rename to final dir with Chinese path
    _TMP = r'C:\Users\28100\AppData\Local\Temp\opencode'
    os.makedirs(_TMP, exist_ok=True)
    for fmt in ['pdf', 'png']:
        tmp_dst = os.path.join(_TMP, f'{name}.{fmt}')
        fig.savefig(tmp_dst, dpi=300 if fmt == 'png' else None, bbox_inches='tight', pad_inches=0.05)
        final_dst = os.path.join(_FINAL_DIR, f'{name}.{fmt}')
        if os.path.exists(final_dst):
            os.remove(final_dst)
        os.rename(tmp_dst, final_dst)
    plt.close(fig); print(f'  Saved {name}')





# ─────────────────────── Fig 0: Model Architecture Diagram ───────────────────────
def fig_mtkan_architecture():
    """Schematic of MT-KAN: KAN front-end -> shared MLP trunk -> 5 task heads."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    fig, ax = plt.subplots(figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.42))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis('off')

    def box(x, y, w, h, label, fc='#EAF2F8', ec='#0072B2', fs=6, lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                                    facecolor=fc, edgecolor=ec, linewidth=lw))
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fs)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle='-|>', mutation_scale=6, lw=0.7, color='#555555'))

    # Input nodes
    in_labels = ['C$_2$H$_2$ flow', 'Bias voltage', 'Dep. time', 'Material']
    ys_in = np.linspace(31, 9, 4)
    for y, lab in zip(ys_in, in_labels):
        ax.add_patch(Circle((6, y), 1.6, facecolor='white', edgecolor='#333333', lw=0.8))
        ax.text(6, y - 3.1, lab, ha='center', va='top', fontsize=8.2)

    # KAN layer (drawn as block with spline edge motif)
    ax.add_patch(FancyBboxPatch((14, 6), 12, 30, boxstyle='round,pad=0.4',
                                facecolor='#D6EAF8', edgecolor='#0072B2', linewidth=1.0))
    ax.text(20, 21, 'KANLinear\n4$\\rightarrow$64', ha='center', va='center',
            fontsize=9.0, fontweight='bold')
    ax.text(20, 15.5, 'learnable B-spline\n(grid=20, $k$=3)\n+ SiLU residual', ha='center',
            va='center', fontsize=6.9, color='#333333')
    # wavy edges input->KAN
    xx = np.linspace(7.8, 13.6, 40)
    for y in ys_in:
        yy = y + 1.1 * np.sin(np.linspace(0, 3 * np.pi, 40))
        ax.plot(xx, yy, color='#0072B2', lw=0.45, alpha=0.75)

    box(30, 17, 9, 8, 'LayerNorm\nSiLU', fc='#FEF9E7', ec='#B7950B')
    arrow(26.6, 21, 29.6, 21)

    # MLP trunk
    trunk_w = [10, 10, 10]
    trunk_x = [43, 56, 69]
    for wx, tx in zip(trunk_w, trunk_x):
        box(tx, 17, wx, 8, '', fc='#EAF2F8', ec='#2874A6')
        ax.text(tx + wx / 2, 22.6, 'Linear', ha='center', va='center', fontsize=7.8)
        ax.text(tx + wx / 2, 19.6, 'LN + SiLU', ha='center', va='center', fontsize=6.6)
    dims = ['64$\\rightarrow$48', '48$\\rightarrow$32', '32$\\rightarrow$16']
    for d, tx in zip(dims, trunk_x):
        ax.text(tx + 5, 15.6, d, ha='center', va='top', fontsize=7.2, color='#333333')
    arrow(39.4, 21, 42.6, 21); arrow(53.4, 21, 55.6, 21); arrow(66.4, 21, 68.6, 21)

    # Task heads
    head_labels = ['$S_a$ (nm)', '$H$ (GPa)', '$E$ (GPa)', r'$\sigma$ (GPa)', '$H/E^*$']
    xs_head = np.linspace(84, 84, 5)
    ys_head = np.linspace(33, 5, 5)
    for yh, hl in zip(ys_head, head_labels):
        box(82, yh - 2.6, 11, 5.2, '16$\\rightarrow$32$\\rightarrow$1', fs=4.6)
        ax.text(95, yh, hl, ha='left', va='center', fontsize=8.7)
        arrow(79.4, 21, 81.6, yh)

    # Stage captions
    ax.text(20, 38.5, 'KAN front-end', ha='center', fontsize=9.8, fontweight='bold', color='#0072B2')
    ax.text(57.5, 27.5, 'Shared MLP backend', ha='center', fontsize=9.8,
            fontweight='bold', color='#2874A6')
    ax.text(87.5, 38.5, 'Task heads ($\\times$5)', ha='center', fontsize=9.8,
            fontweight='bold', color='#1A5276')
    ax.text(50, 0.8, 'Total parameters: 14,629   |   Physics-informed regularisation on shared representation',
            ha='center', fontsize=7.8, style='italic', color='#555555')
    plt.tight_layout()
    save_figure(fig, 'fig0_architecture')


# ─────────────────────── Fig 1: Parity Plots (per task, by source) ───────────────────────
def fig_parity_plots(preds, targets, mask_test, source_labels, task_names):
    task_short = ['$S_a$', '$H$', '$E$', r'$\sigma$', '$H/E^*$']
    unit = ['nm', 'GPa', 'GPa', 'GPa', '']
    src_colors = {'exact': C7[0], 'synthetic': C7[1], 'digitized': C7[3]}
    fig, axes = plt.subplots(1, len(task_names),
                             figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.34))
    metrics_json = None
    res_path = os.path.join(_FINAL_DIR, '..', 'results', 'results_summary.json')
    try:
        import json
        with open(res_path) as f:
            metrics_json = json.load(f)['MT-KAN']
    except Exception:
        pass
    for t, ax in enumerate(axes):
        valid = mask_test[:, t].astype(bool)
        yt, yp = targets[valid, t], preds[valid, t]
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        pad = 0.06 * (lims[1] - lims[0]); lims = [lims[0] - pad, lims[1] + pad]
        ax.plot(lims, lims, ls='--', c='grey', lw=0.6, zorder=1)
        for src in ['exact', 'synthetic', 'digitized']:
            m = source_labels.get(src)
            if m is None:
                continue
            sel = valid & m
            if sel.sum() > 0:
                ax.scatter(targets[sel, t], preds[sel, t], s=4, alpha=0.55,
                           c=src_colors[src], edgecolors='none', label=src, zorder=2)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        ax.text(0.04, 0.94, f'$R^2$={r2:.3f}\nRMSE={rmse:.2f}', transform=ax.transAxes,
                va='top', fontsize=7.8,
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.5))
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(task_short[t] + (f' ({unit[t]})' if unit[t] else ''), fontsize=9.8)
        if t == 0:
            ax.set_ylabel('Predicted'); ax.legend(frameon=False, fontsize=6.9,
                                                  loc='lower right', handletextpad=0.1)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel('Measured')
        ax.tick_params(labelsize=5.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'fig1_parity_plots')


# ─────────────────────── Fig 3: Training Curves ───────────────────────
def fig_training_curves(history_dict):
    styles = {'MT-KAN': (C7[0], '-'), 'MLP': (C7[1], '-'), 'KAN (no physics)': (C7[2], '--')}
    disp = {'MT-KAN': 'MT-KAN', 'MLP': 'MLP', 'KAN (no physics)': 'KAN w/o physics'}
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.36))
    for name, hist in history_dict.items():
        if name not in styles:
            continue
        c, ls = styles[name]
        ep = np.arange(1, len(hist['train_loss']) + 1)
        axes[0].plot(ep, hist['train_loss'], color=c, ls=ls, lw=0.8, label=disp[name])
        axes[1].plot(ep, hist['val_loss'], color=c, ls=ls, lw=0.8)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Training loss')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation loss')
    axes[0].legend(frameon=False, fontsize=8.2)
    for ax in axes:
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'fig3_training_curves')


# ─────────────────────── Fig 5: Ablation Study (2x2) ───────────────────────
ABLATION_VARIANTS = ['physics + weights', 'physics only', 'weights only', 'neither']
_ABL_STYLE = {
    'physics + weights': (C7[0], 'MT-KAN (full)'),
    'physics only':      (C7[2], 'physics, no task weights'),
    'weights only':      (C7[1], 'task weights, no physics'),
    'neither':           ('#999999', 'neither'),
}


def fig_ablation_study(ablation_metrics, task_names):
    """2x2 factorial ablation: physics prior on/off x task weights on/off."""
    variants = [v for v in ABLATION_VARIANTS if v in ablation_metrics]
    all_tasks = list(task_names) + ['Avg']
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE[0] * 2.05, NATURE_SINGLE[0] * 1.0))
    x = np.arange(len(all_tasks)); n = max(len(variants), 1); width = 0.8 / n
    for vi, v in enumerate(variants):
        color, label = _ABL_STYLE[v]
        vals = [ablation_metrics[v][t]['R2'] if t in ablation_metrics[v] else 0
                for t in task_names]
        vals.append(float(np.mean(vals)))
        off = (vi - (n - 1) / 2) * width
        bars = ax.bar(x + off, vals, width, label=label, color=color,
                      alpha=0.88, edgecolor='white', linewidth=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=6.4, rotation=90)
    ax.axvline(x=len(task_names) - 0.5, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['$S_a$', '$H$', '$E$', r'$\sigma$', '$H/E^*$', '\nAvg'], fontsize=9.8)
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.set_ylabel('$R^2$ Score'); ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc='upper center', fontsize=7.6, ncol=2)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'fig5_ablation')


# ─────────────────────── Fig 10: Gradient-based Feature Importance ───────────────────────
def fig_feature_importance(model, X_test, mask_test, task_names, feature_names):
    device = next(model.parameters()).device
    model.eval()
    imp = np.zeros((len(task_names), len(feature_names)))
    feat_labels = ['C$_2$H$_2$\nflow', 'Bias\nvoltage', 'Dep.\ntime', 'Material\ntype']
    task_short = ['$S_a$', '$H$', '$E$', r'$\sigma$', '$H/E^*$']
    with torch.no_grad():
        pass
    for t in range(len(task_names)):
        rows = mask_test[:, t].astype(bool)
        if rows.sum() < 5:
            continue
        x = torch.FloatTensor(X_test[rows]).to(device).requires_grad_(True)
        out = model(x)
        grad = torch.autograd.grad(out[:, t].sum(), x)[0].abs().mean(0).cpu().numpy()
        imp[t] = grad / max(grad.sum(), 1e-10)
    fig, ax = plt.subplots(figsize=(NATURE_SINGLE[0] * 1.35, NATURE_SINGLE[0]))
    sns.heatmap(imp, ax=ax, cmap='YlGnBu', annot=True, fmt='.2f',
                annot_kws={'fontsize': 5.5}, xticklabels=feat_labels,
                yticklabels=task_short, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Normalised |gradient| importance'})
    ax.tick_params(labelsize=6)
    plt.tight_layout()
    save_figure(fig, 'fig10_feature_importance')


# ─────────────────────── Fig 4: Model Comparison (R² bar chart) ───────────────────────
def fig_model_comparison(all_metrics, model_names, task_names):
    n_models = len(model_names); n_tasks = len(task_names)
    all_tasks = list(task_names) + ['Avg']; n_cols = len(all_tasks)
    fig, ax = plt.subplots(1, 1, figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.75))
    x = np.arange(n_cols)
    width = min(0.8 / n_models, 0.12)
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)
    for m_idx, name in enumerate(model_names):
        metrics = all_metrics[name]
        r2_per_task = [metrics[t]['R2'] if t in metrics else 0 for t in task_names]
        avg_r2 = np.mean(r2_per_task)
        r2_vals = r2_per_task + [avg_r2]
        bars = ax.bar(x + offsets[m_idx], r2_vals, width, label=name,
                      color=C7[m_idx % len(C7)], alpha=0.85, edgecolor='white', linewidth=0.3)
        for bar, val in zip(bars, r2_vals):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=6.8, rotation=90)
        bars[-1].set_edgecolor('black'); bars[-1].set_linewidth(0.8)
    ax.set_xticks(x); ax.set_xticklabels(all_tasks, rotation=20, ha='right')
    ax.get_xticklabels()[-1].set_fontweight('bold')
    ax.set_ylabel('$R^2$ Score'); ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, loc='upper right', ncol=2, fontsize=8.2)
    ax.axhline(y=0, color='grey', lw=0.5)
    ax.axvline(x=n_tasks - 0.5, color='grey', lw=0.5, ls='--', alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'fig4_model_comparison')

# ─────────────────────── Fig 11: RMSE + MAE Heatmap ───────────────────────
def fig_rmse_mae_heatmap(all_metrics, model_names, task_names):
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.55))
    model_short = {
        'MT-KAN': 'KAN', 'MLP': 'MLP', 'Random Forest': 'RF',
        'XGBoost': 'XGB', 'SVR': 'SVR', 'kNN': 'kNN', 'ElasticNet': 'ENet'
    }
    short_names = [model_short.get(m, m) for m in model_names]
    task_short = ['$S_a$', '$H$', '$E$', r'$\sigma$', '$H/E^*$']
    for ax, metric, fmt, cmap, label in [
        (axes[0], 'RMSE', '.2f', 'Blues', 'RMSE'),
        (axes[1], 'MAE', '.2f', 'Oranges', 'MAE')]:
        data = np.array([[all_metrics[m][t][metric] if t in all_metrics[m] else 0
                          for t in task_names] for m in model_names])
        # Normalize per task: divide by task mean so values near 1 = average
        task_mean = data.mean(axis=0, keepdims=True)
        task_mean[task_mean == 0] = 1
        data_norm = data / task_mean
        sns.heatmap(data_norm, ax=ax, cmap=cmap, annot=data, fmt=fmt,
                    xticklabels=task_short, yticklabels=short_names,
                    vmin=0, vmax=data_norm.max() * 1.05,
                    cbar_kws={'label': f'Relative {metric}'},
                    linewidths=0.5, linecolor='white')
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel(''); ax.set_ylabel('')
    plt.tight_layout()
    save_figure(fig, 'fig11_rmse_mae_heatmap')





# ─────────────────────── Fig 6: Parameter Maps (2 rows, cleaner) ───────────────────────
def fig_parameter_maps(model, ds, task_names, save_name='fig6_parameter_maps'):
    flow_grid = np.linspace(0, 100, 100)
    bias_grid = np.linspace(50, 800, 80)
    FF, BB = np.meshgrid(flow_grid, bias_grid)
    n_tasks = len(task_names)
    norm = ds.get_normalization()
    grid_points = np.zeros((*FF.shape, 4))
    for i in range(FF.shape[0]):
        for j in range(FF.shape[1]):
            grid_points[i, j, 0] = (FF[i, j] - norm['feat_mean'][0]) / norm['feat_std'][0]
            grid_points[i, j, 1] = (BB[i, j] - norm['feat_mean'][1]) / norm['feat_std'][1]
            grid_points[i, j, 2] = (600 - norm['feat_mean'][2]) / norm['feat_std'][2]
            grid_points[i, j, 3] = 0
    x_flat = grid_points.reshape(-1, 4)
    device = next(model.parameters()).device
    with torch.no_grad():
        model.eval()
        x_tensor = torch.FloatTensor(x_flat).to(device)
        pred_all = model(x_tensor).cpu().numpy()

    task_short = ['$S_a$ (nm)', '$H$ (GPa)', '$E$ (GPa)', r'$\sigma$ (GPa)', '$H/E^*$']
    fig, axes = plt.subplots(2, 3, figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.55))
    axes = axes.flatten()
    for t in range(n_tasks):
        ax = axes[t]
        Z = pred_all[:, t].reshape(FF.shape)
        Z_denorm = Z * norm['tgt_std'][t] + norm['tgt_mean'][t]
        levels = 15
        cf = ax.contourf(FF, BB, Z_denorm, levels=levels, cmap='viridis', alpha=0.85)
        ax.contour(FF, BB, Z_denorm, levels=levels, colors='k', linewidths=0.15, alpha=0.25)
        cbar = plt.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(task_short[t], fontsize=9.0); cbar.ax.tick_params(labelsize=5)
        ax.set_xlabel('C$_2$H$_2$ flow (sccm)'); ax.set_ylabel('Bias (V)')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    axes[5].set_visible(False)
    plt.tight_layout()
    save_figure(fig, save_name)


# ─────────────────────── Fig 7: KAN Spline Coefficient Heatmap ───────────────────────
def fig_spline_visualization(model, feature_names, save_name='fig7_spline_functions'):
    """Spline coefficient heatmap: rows=output neurons, cols=basis fn per input feature."""
    kan_layer = model.backbone[0]
    with torch.no_grad():
        coeffs = kan_layer.spline_coeffs.cpu().numpy()
    n_out = coeffs.shape[0]; n_in = coeffs.shape[1]; n_basis = coeffs.shape[2]
    feat_labels = ['C$_2$H$_2$', 'Bias', 'Time', 'Material']
    fig, axes = plt.subplots(1, n_in, figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.30))
    vmax = max(abs(coeffs.min()), abs(coeffs.max()))
    for i in range(n_in):
        ax = axes[i]
        sns.heatmap(coeffs[:, i, :], ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=False, yticklabels=False, vmin=-vmax, vmax=vmax,
                    cbar_kws={'shrink': 0.8}, linewidths=0, linecolor='white')
        ax.set_title(f'{feat_labels[i]}', fontsize=10.5)
        ax.set_xlabel('Basis index'); ax.set_ylabel('Neuron')
    plt.tight_layout()
    save_figure(fig, save_name)

 
 # ─────────────────────── Fig 8: Correlation Matrix ───────────────────────
def fig_correlation_matrix(all_preds_dict, all_targets, task_names, model_names):
    n_tasks = len(task_names)
    fig, axes = plt.subplots(1, 2, figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] / 2.0))
    valid = ~np.isnan(all_targets).any(axis=1)
    for ax, data, title in [
        (axes[0], all_targets[valid], 'Ground Truth'),
        (axes[1], all_preds_dict.get('MT-KAN', all_targets)[valid], 'MT-KAN Predicted')]:
        corr = np.corrcoef(data.T)
        sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0,
                    xticklabels=task_names, yticklabels=task_names,
                    vmin=-1, vmax=1, cbar_kws={'shrink': 0.8}, annot=True,
                    fmt='.2f', annot_kws={'fontsize': 5})
        ax.set_title(title, fontsize=10.5)
    plt.tight_layout()
    save_figure(fig, 'fig8_correlation_matrix')








# ─────────────────────── Fig 13: t-SNE of Feature Space ───────────────────────
def fig_tsne_visualization(ds, task_names):
    """t-SNE projection of feature space, colored by source and key targets."""
    from sklearn.manifold import TSNE
    X = ds.features
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, init='pca')
    Z = tsne.fit_transform(X)
    df = ds.df
    sources = df['source'].values
    src_colors = {'exact': C7[0], 'synthetic': C7[1], 'digitized': C7[3]}
    fig, axes = plt.subplots(1, 3, figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.4))
    # Panel 1: by source
    ax = axes[0]
    for src in ['exact', 'synthetic', 'digitized']:
        mask = sources == src
        if mask.sum() > 5:
            ax.scatter(Z[mask, 0], Z[mask, 1], s=3, alpha=0.4,
                      color=src_colors[src], label=src, edgecolors='none')
    ax.legend(frameon=False, fontsize=7.5); ax.set_title('By data source', fontsize=9.0)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Panel 2: by Sa (roughness)
    ax = axes[1]
    valid_sa = ~np.isnan(df['Sa_nm'].values)
    sc = ax.scatter(Z[valid_sa, 0], Z[valid_sa, 1], s=3, alpha=0.5,
                   c=df['Sa_nm'].values[valid_sa], cmap='viridis', edgecolors='none')
    plt.colorbar(sc, ax=ax, shrink=0.8).set_label('$S_a$ (nm)', fontsize=7.5)
    ax.set_title('By $S_a$ roughness', fontsize=9.0)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # Panel 3: by H (hardness)
    ax = axes[2]
    valid_h = ~np.isnan(df['H_GPa'].values)
    sc = ax.scatter(Z[valid_h, 0], Z[valid_h, 1], s=3, alpha=0.5,
                   c=df['H_GPa'].values[valid_h], cmap='viridis', edgecolors='none')
    plt.colorbar(sc, ax=ax, shrink=0.8).set_label('$H$ (GPa)', fontsize=7.5)
    ax.set_title('By $H$ hardness', fontsize=9.0)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'fig13_tsne')


# ─────────────────────── Fig 15: Violin Plot of Targets by Source ───────────────────────
def fig_violin_by_source(ds, task_names):
    """Violin plots showing distribution of each target, split by data source."""
    df = ds.df
    task_short = ['$S_a$ (nm)', '$H$ (GPa)', '$E$ (GPa)', r'$\sigma$ (GPa)', '$H/E^*$']
    src_palette = ['#0072B2', '#009E73', '#CC79A7']  # blue, green, pink for exact/synthetic/digitized
    fig, axes = plt.subplots(1, len(task_names), figsize=(NATURE_DOUBLE[0], NATURE_DOUBLE[0] * 0.45))
    for t, (task, short) in enumerate(zip(task_names, task_short)):
        ax = axes[t]
        src_data = []
        src_labels_local = []
        for src in ['exact', 'synthetic', 'digitized']:
            vals = df.loc[df['source'] == src, task].dropna().values
            if len(vals) > 3:
                src_data.append(vals)
                src_labels_local.append(f'{src}\n(n={len(vals)})')
        if src_data:
            parts = ax.violinplot(src_data, positions=range(len(src_data)),
                                  showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(src_palette[i]); pc.set_alpha(0.5)
            parts['cmeans'].set_color('k'); parts['cmedians'].set_color('grey')
            ax.set_xticks(range(len(src_data)))
            ax.set_xticklabels(src_labels_local, fontsize=7.5)
        ax.set_title(short, fontsize=9.0); ax.set_ylabel('')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_figure(fig, 'fig15_violin_source')


# ─────────────────────── Main ───────────────────────
def generate_all_figures(all_metrics, all_preds_dict, all_targets, history_dict,
                         model_names, task_names, feature_names, model=None, ds=None,
                         source_labels=None, ablation_metrics=None, save_dir=None,
                         mask_test=None, X_test=None):
    if save_dir is not None: set_save_dir(save_dir)
    print('Generating figures...')
    fig_mtkan_architecture()
    fig_model_comparison(all_metrics, model_names, task_names)
    if model is not None and ds is not None: fig_parameter_maps(model, ds, task_names)
    if model is not None: fig_spline_visualization(model, feature_names)
    fig_correlation_matrix(all_preds_dict, all_targets, task_names, model_names)
    fig_rmse_mae_heatmap(all_metrics, model_names, task_names)
    if ds is not None:
        fig_tsne_visualization(ds, task_names)
        fig_violin_by_source(ds, task_names)
    if mask_test is not None:
        fig_parity_plots(all_preds_dict['MT-KAN'], all_targets, mask_test,
                         source_labels or {}, task_names)
        if model is not None and X_test is not None:
            fig_feature_importance(model, X_test, mask_test, task_names, feature_names)
    if history_dict:
        fig_training_curves(history_dict)
    if ablation_metrics:
        fig_ablation_study(ablation_metrics, task_names)
    print(f'\nAll figures saved to {_FINAL_DIR}/')

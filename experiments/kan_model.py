"""
Multi-Task Kolmogorov-Arnold Network (MT-KAN) implementation.
B-spline based learnable activation functions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BSplineBasis(nn.Module):
    """B-spline basis functions (batched over features)."""
    def __init__(self, grid_min=-1., grid_max=1., grid_size=8, spline_order=3):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_max - grid_min) / grid_size
        knots = torch.linspace(grid_min - spline_order * h,
                                grid_max + spline_order * h,
                                grid_size + 2 * spline_order + 1)
        self.register_buffer('knots', knots)

    def forward(self, x):
        x = x.squeeze(-1)
        bases = ((x.unsqueeze(-1) >= self.knots[:-1]) &
                 (x.unsqueeze(-1) < self.knots[1:])).float()
        for k in range(1, self.spline_order + 1):
            left_num = x.unsqueeze(-1) - self.knots[:-(k+1)]
            left_den = (self.knots[k:-1] - self.knots[:-(k+1)]) + 1e-10
            right_num = self.knots[k+1:] - x.unsqueeze(-1)
            right_den = (self.knots[k+1:] - self.knots[1:-k]) + 1e-10
            left = (left_num / left_den) * bases[..., :-1]
            right = (right_num / right_den) * bases[..., 1:]
            bases = left + right
        return bases


class KANLinear(nn.Module):
    """KAN linear layer with efficient batched forward pass."""
    def __init__(self, in_features, out_features, grid_size=8, spline_order=3,
                 residual_std=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        n_basis = grid_size + spline_order

        self.spline_coeffs = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.1)
        self.residual_weight = nn.Parameter(torch.randn(out_features, in_features) * residual_std)
        self.scale = nn.Parameter(torch.ones(out_features))

        self.basis_fn = BSplineBasis(grid_min=-1., grid_max=1.,
                                      grid_size=grid_size, spline_order=spline_order)
        self._init_weights()

    def _init_weights(self):
        active = self.spline_order + 1
        fan = self.in_features * active + self.out_features
        std = np.sqrt(2.0 / fan)
        with torch.no_grad():
            self.spline_coeffs.normal_(0, std)
            self.residual_weight.normal_(0, std)
            nn.init.ones_(self.scale)

    def forward(self, x):
        x_clamped = torch.tanh(x * 1.2)  # smooth clamp with tanh, slight over-range
        bases = self.basis_fn(x_clamped.unsqueeze(-1))
        b = bases.reshape(x.shape[0], -1)
        c = self.spline_coeffs.reshape(self.out_features, -1)
        spline_out = b @ c.T
        residual_out = F.silu(x_clamped) @ self.residual_weight.T
        output = (spline_out + residual_out) * self.scale[None, :]
        return output


class KAN(nn.Module):
    """KAN model with multiple layers, norms, and residual connections."""
    def __init__(self, layer_dims, grid_size=8, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                KANLinear(layer_dims[i], layer_dims[i+1],
                         grid_size=grid_size, spline_order=spline_order)
            )
            self.norms.append(nn.LayerNorm(layer_dims[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            residual = x if x.shape[-1] == self.layers[i].out_features else None
            x = layer(x)
            x = self.norms[i](x)
            if i < len(self.layers) - 1:
                x = F.silu(x)
            if residual is not None:
                x = x + residual
        return x


class MultiTaskKAN(nn.Module):
    """
    Multi-Task KAN: wide KAN front-end + deep MLP backend + task-specific heads.
    """
    def __init__(self, n_input=4, n_hidden=64, n_tasks=5,
                 grid_size=20, spline_order=3):
        super().__init__()
        self.n_tasks = n_tasks

        # Single rich KAN layer + deeper MLP backend
        h1 = max(8, n_hidden)
        h2 = max(8, n_hidden * 3 // 4)
        h3 = max(4, n_hidden // 2)
        h4 = max(4, n_hidden // 4)

        self.backbone = nn.Sequential(
            KANLinear(n_input, h1, grid_size=grid_size, spline_order=spline_order),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Linear(h2, h3),
            nn.LayerNorm(h3),
            nn.SiLU(),
            nn.Linear(h3, h4),
            nn.LayerNorm(h4),
            nn.SiLU(),
        )

        h_out = h4
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h_out, max(4, h_out * 2)),
                nn.SiLU(),
                nn.Linear(max(4, h_out * 2), 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x, task_idx=None):
        shared = self.backbone(x)
        if task_idx is not None:
            return self.heads[task_idx](shared).squeeze(-1)
        return torch.cat([h(shared) for h in self.heads], dim=-1)

    def get_spline_functions(self, x_grid):
        self.eval()
        with torch.no_grad():
            x_zeros = torch.zeros(x_grid.shape[0], 4, device=next(self.parameters()).device)
            shared = self.backbone(x_zeros)
        return shared


class MLPBaseline(nn.Module):
    """
    Capacity-matched MLP baseline.

    Three hidden layers (128 -> 64 -> 32) with LayerNorm + SiLU, followed by
    per-task heads of the same shape pattern as MT-KAN (32 -> 16 -> 1).
    Total 14,149 parameters, close to MT-KAN's 14,629, so the comparison
    isolates the effect of the B-spline front-end rather than model capacity.

    Note: the attribute holding the trunk is deliberately named ``shared`` and
    not ``backbone``; ``train_epoch`` gates the physics penalty on
    ``hasattr(model, 'backbone')`` so that the baseline stays physics-free.
    """
    def __init__(self, n_input=4, n_tasks=5, widths=(128, 64, 32), head_mult=2):
        super().__init__()
        self.n_tasks = n_tasks

        layers = []
        prev = n_input
        for w in widths:
            layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.SiLU()]
            prev = w
        self.shared = nn.Sequential(*layers)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev, prev * head_mult // 4),
                nn.SiLU(),
                nn.Linear(prev * head_mult // 4, 1),
            )
            for _ in range(n_tasks)
        ])

    def forward(self, x, task_idx=None):
        shared = self.shared(x)
        if task_idx is not None:
            return self.heads[task_idx](shared).squeeze(-1)
        return torch.cat([h(shared) for h in self.heads], dim=-1)

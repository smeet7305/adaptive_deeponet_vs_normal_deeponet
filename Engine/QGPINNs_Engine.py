import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import gamma
import math
import copy
import os
import random
import csv
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import solve_banded

MATRIX_DTYPE = torch.float64
NET_DTYPE = torch.float64
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_rows(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class FourierEmbedding(nn.Module):
    def __init__(self, in_dim=2, embed_dim=64, sigma=1.0, sampling='sobol'):
        super().__init__()
        n_freqs = embed_dim // 2
        if sampling == 'sobol':
            sobol = torch.quasirandom.SobolEngine(dimension=in_dim, scramble=True)
            u = sobol.draw(n_freqs).to(NET_DTYPE)
            u = torch.clamp(u, 1e-06, 1 - 1e-06)
            B = torch.erfinv(2 * u - 1) * 2 ** 0.5 * sigma
            B = B.T
        else:
            B = torch.randn(in_dim, n_freqs, dtype=NET_DTYPE) * sigma
        self.register_buffer('B', B)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class PINN_Net(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, hidden_layers=3, use_fourier=False, fourier_dim=64, fourier_sigma=1.0,
                 fourier_sampling='sobol'):
        super().__init__()
        self.use_fourier = use_fourier
        if use_fourier:
            self.fourier = FourierEmbedding(in_dim, fourier_dim, fourier_sigma, fourier_sampling)
            first_in = fourier_dim
        else:
            self.fourier = None
            first_in = in_dim
        layers = [nn.Linear(first_in, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_fourier:
            x = self.fourier(x)
        return self.net(x)


def _caputo_power_term(c, xi, order, t):
    xi_t = xi if torch.is_tensor(xi) else torch.tensor(float(xi), dtype=t.dtype, device=t.device)
    order_t = order if torch.is_tensor(order) else torch.tensor(float(order), dtype=t.dtype, device=t.device)
    order_eff = torch.minimum(order_t, xi_t)
    log_ratio = torch.lgamma(xi_t + 1.0) - torch.lgamma(xi_t - order_eff + 1.0)
    ratio = torch.exp(log_ratio)
    t_pow = t.clamp(min=1e-12).pow(xi_t - order_eff)
    return c * ratio * t_pow


class SplitPINN_Net(nn.Module):
    def __init__(self, smooth_in_dim, hidden_dim=64, hidden_layers=3, use_fourier=False, fourier_dim=64,
                 fourier_sigma=1.0, fourier_sampling='sobol'):
        super().__init__()
        self.smooth = PINN_Net(in_dim=smooth_in_dim, hidden_dim=hidden_dim, hidden_layers=hidden_layers,
                               use_fourier=use_fourier, fourier_dim=fourier_dim, fourier_sigma=fourier_sigma,
                               fourier_sampling=fourier_sampling)
        self.use_fourier = use_fourier
        spatial_in_dim = smooth_in_dim - 1
        if self.use_fourier:
            self.spatial_fourier = FourierEmbedding(
                in_dim=spatial_in_dim,
                embed_dim=fourier_dim,
                sigma=fourier_sigma,
                sampling=fourier_sampling
            )
            coeff_in_dim = fourier_dim
        else:
            self.spatial_fourier = None
            coeff_in_dim = spatial_in_dim
        layers = [nn.Linear(coeff_in_dim, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.coeff = nn.Sequential(*layers)
        for lyr in self.coeff:
            if isinstance(lyr, nn.Linear):
                nn.init.xavier_uniform_(lyr.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(lyr.bias)

    def forward_components(self, smooth_inp, x_spatial):
        u_smooth = self.smooth(smooth_inp)
        if self.use_fourier:
            x_embedded = self.spatial_fourier(x_spatial)
            c = self.coeff(x_embedded)
        else:
            c = self.coeff(x_spatial)
        return u_smooth, c


class SDCNet(nn.Module):

    def __init__(self, smooth_dim, raw_dim, hidden_dim=64, hidden_layers=3, use_fourier=False, fourier_dim=64,
                 fourier_sigma=1.0, fourier_sampling='sobol'):
        super().__init__()
        self.use_fourier = use_fourier
        self.raw_dim = raw_dim
        if use_fourier:
            self.fourier = FourierEmbedding(smooth_dim, fourier_dim, fourier_sigma, fourier_sampling)
            smooth_out_dim = fourier_dim
        else:
            self.fourier = None
            smooth_out_dim = smooth_dim
        first_in = smooth_out_dim + raw_dim
        layers = [nn.Linear(first_in, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

    def forward(self, x_smooth, x_raw):
        if self.use_fourier:
            x_smooth = self.fourier(x_smooth)
        inp = torch.cat([x_smooth, x_raw], dim=-1)
        return self.net(inp)
    
def generate_random_function(x, n_modes=10):
    coeffs = torch.randn(n_modes, dtype=NET_DTYPE)

    result = torch.zeros_like(x)

    for k in range(n_modes):
        result += coeffs[k] * torch.sin(
            (k + 1) * math.pi * x
        )

    return result
    
class DeepONet(nn.Module):
    def __init__(
        self,
        branch_dim,
        trunk_dim=2,
        hidden_dim=128,
        hidden_layers=4,
        latent_dim=128,
        use_fourier=True,
        fourier_dim=128,
        fourier_sigma=1.0
    ):
        super().__init__()

        # -------------------------
        # Branch network
        # -------------------------
        branch_layers = [
            nn.Linear(branch_dim, hidden_dim),
            nn.Tanh()
        ]

        for _ in range(hidden_layers - 1):
            branch_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ]

        branch_layers.append(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.branch = nn.Sequential(*branch_layers)

        # -------------------------
        # Trunk network
        # -------------------------
        self.use_fourier = use_fourier

        if use_fourier:
            self.trunk_fourier = FourierEmbedding(
                in_dim=trunk_dim,
                embed_dim=fourier_dim,
                sigma=fourier_sigma
            )
            trunk_input_dim = fourier_dim
        else:
            self.trunk_fourier = None
            trunk_input_dim = trunk_dim

        trunk_layers = [
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.Tanh()
        ]

        for _ in range(hidden_layers - 1):
            trunk_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ]

        trunk_layers.append(
            nn.Linear(hidden_dim, latent_dim)
        )

        self.trunk = nn.Sequential(*trunk_layers)

        self.bias = nn.Parameter(
            torch.zeros(1, dtype=NET_DTYPE)
        )

    def forward(self, branch_input, trunk_input):
        b = self.branch(branch_input)
        if self.use_fourier:
            trunk_input = self.trunk_fourier(trunk_input)
        t = self.trunk(trunk_input)
        if b.ndim == 1:
            b = b.unsqueeze(0)
        if b.shape[0] != t.shape[0]:
            b = b.expand(t.shape[0], -1)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias


class PretrainedDeepONet(nn.Module):
    """
    Physics-Informed DeepONet using trained & frozen weights and biases from training_weights_biases/.
    Architecture:
      - Branch Network: Modified MLP (Wang et al. 2021) with 7 hidden layers of width 200, input dim 304,
                        and linear transformation projections U and V.
      - Trunk Network: Fourier Feature MLP with 6 hidden layers of width 200, input dim 2 (t, x),
                       embedded via 100 random Fourier frequencies.
    """
    def __init__(self, model_type='inner', weights_dict=None, weights_dir=None, u_branch=None):
        super().__init__()
        self.model_type = str(model_type).lower()
        if weights_dir is None:
            cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training_weights_biases')
            if os.path.exists(cand):
                weights_dir = cand
            else:
                weights_dir = 'training_weights_biases'
        self.weights_dir = weights_dir

        # Load Fourier random frequencies
        ff_path = os.path.join(weights_dir, 'trunk_fourier_frequencies.npy')
        if os.path.exists(ff_path):
            FF_np = np.load(ff_path)
        else:
            FF_np = np.random.RandomState(0).randn(2, 100).astype(np.float32) * 5.0
        self.register_buffer('FF', torch.tensor(FF_np, dtype=torch.float32))

        # Branch input buffer (304 sensor/velocity values)
        if u_branch is not None:
            if not torch.is_tensor(u_branch):
                u_branch = torch.tensor(u_branch, dtype=torch.float32)
            if u_branch.ndim == 1:
                u_branch = u_branch.unsqueeze(0)
            self.register_buffer('u_branch', u_branch)
        else:
            self.register_buffer('u_branch', torch.zeros((1, 304), dtype=torch.float32))

        self._build_layers()

        if weights_dict is not None:
            self.load_weights_dict(weights_dict)
        elif weights_dir is not None and os.path.exists(weights_dir):
            self.load_pretrained_from_dir(weights_dir, self.model_type)

    def _build_layers(self):
        m = 304
        w = 200
        # Branch modified MLP parameters
        self.branch_U_w = nn.Parameter(torch.zeros(m, w, dtype=torch.float32))
        self.branch_U_b = nn.Parameter(torch.zeros(w, dtype=torch.float32))
        self.branch_V_w = nn.Parameter(torch.zeros(m, w, dtype=torch.float32))
        self.branch_V_b = nn.Parameter(torch.zeros(w, dtype=torch.float32))

        self.branch_layers_w = nn.ParameterList([
            nn.Parameter(torch.zeros(m if i == 0 else w, w, dtype=torch.float32))
            for i in range(7)
        ])
        self.branch_layers_b = nn.ParameterList([
            nn.Parameter(torch.zeros(w, dtype=torch.float32))
            for i in range(7)
        ])

        # Trunk Fourier Feature MLP parameters
        self.trunk_layers_w = nn.ParameterList([
            nn.Parameter(torch.zeros(w, w, dtype=torch.float32))
            for i in range(6)
        ])
        self.trunk_layers_b = nn.ParameterList([
            nn.Parameter(torch.zeros(w, dtype=torch.float32))
            for i in range(6)
        ])

    def load_weights_dict(self, weights):
        with torch.no_grad():
            self.branch_U_w.copy_(torch.as_tensor(weights['branch_U_weight'], dtype=torch.float32))
            self.branch_U_b.copy_(torch.as_tensor(weights['branch_U_bias'], dtype=torch.float32))
            self.branch_V_w.copy_(torch.as_tensor(weights['branch_V_weight'], dtype=torch.float32))
            self.branch_V_b.copy_(torch.as_tensor(weights['branch_V_bias'], dtype=torch.float32))

            for i in range(7):
                self.branch_layers_w[i].copy_(torch.as_tensor(weights[f'branch_layer_{i}_weight'], dtype=torch.float32))
                self.branch_layers_b[i].copy_(torch.as_tensor(weights[f'branch_layer_{i}_bias'], dtype=torch.float32))

            for i in range(6):
                self.trunk_layers_w[i].copy_(torch.as_tensor(weights[f'trunk_layer_{i}_weight'], dtype=torch.float32))
                self.trunk_layers_b[i].copy_(torch.as_tensor(weights[f'trunk_layer_{i}_bias'], dtype=torch.float32))

    def load_pretrained_from_dir(self, weights_dir, model_type='inner'):
        all_pt = os.path.join(weights_dir, 'all_edge_models_frozen.pt')
        if os.path.exists(all_pt):
            data = torch.load(all_pt, weights_only=True, map_location='cpu')
            if model_type in data:
                self.load_weights_dict(data[model_type])
                return
        single_pt = os.path.join(weights_dir, f'{model_type}_weights_biases.pt')
        if os.path.exists(single_pt):
            data = torch.load(single_pt, weights_only=True, map_location='cpu')
            self.load_weights_dict(data)
            return
        single_npz = os.path.join(weights_dir, f'{model_type}_weights_biases.npz')
        if os.path.exists(single_npz):
            data = np.load(single_npz)
            self.load_weights_dict({k: data[k] for k in data.files})

    def set_branch_input(self, u_branch):
        if not torch.is_tensor(u_branch):
            u_branch = torch.tensor(u_branch, dtype=torch.float32)
        if u_branch.ndim == 1:
            u_branch = u_branch.unsqueeze(0)
        self.u_branch = u_branch.to(self.branch_U_w.device)

    def forward_branch(self, u=None):
        if u is None:
            u = self.u_branch
        u = u.to(self.branch_U_w.dtype).to(self.branch_U_w.device)
        U = torch.tanh(u @ self.branch_U_w + self.branch_U_b)
        V = torch.tanh(u @ self.branch_V_w + self.branch_V_b)
        h = u
        for i in range(6):
            out = torch.tanh(h @ self.branch_layers_w[i] + self.branch_layers_b[i])
            h = out * U + (1.0 - out) * V
        return h @ self.branch_layers_w[6] + self.branch_layers_b[6]

    def forward_trunk(self, tx):
        tx = tx.to(self.FF.dtype).to(self.FF.device)
        proj = tx @ self.FF
        H = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        for i in range(5):
            H = torch.tanh(H @ self.trunk_layers_w[i] + self.trunk_layers_b[i])
        return H @ self.trunk_layers_w[5] + self.trunk_layers_b[5]

    def forward(self, x, t=None, u=None):
        """
        Evaluate solution u(x, t) on edge.
        Coordinates can be provided as (t, x) combined in x, or as separate x and t tensors.
        """
        if t is not None:
            tx = torch.cat([t, x], dim=-1)
        else:
            tx = x
        B = self.forward_branch(u)
        T = self.forward_trunk(tx)
        if B.ndim == 1:
            B = B.unsqueeze(0)
        if B.shape[0] != T.shape[0]:
            B = B.expand(T.shape[0], -1)
        return torch.sum(B * T, dim=-1, keepdim=True)


class DriftDiffusionPhysics:
    """
    Nonlinear Drift-Diffusion PDE on a metric graph edge:
      dt(rho) = dx(eps * dx(rho) - nu * rho * (1 - rho)) + f_target
      Expanded: dt(rho) - eps * dxx(rho) + nu * (1 - 2*rho) * dx(rho) - f_target = 0
    Flux:
      J = -eps * dx(rho) + nu * rho * (1 - rho)
    """
    def __init__(self, edge_idx=0, eps=0.1, nu=1.0, ic_func=None, source_func=None):
        self.edge_idx = edge_idx
        self.eps = float(eps)
        self.nu = float(nu)
        self.ic_func = ic_func
        self.source_func = source_func

    def F(self, x, t, u, u_x, u_xx, dt_u, f_target=None):
        res = dt_u - self.eps * u_xx + self.nu * (1.0 - 2.0 * u) * u_x
        if f_target is not None:
            res = res - f_target
        return res

    def get_flux(self, u, u_x):
        return -self.eps * u_x + self.nu * u * (1.0 - u)

    def get_ic(self, x):
        if self.ic_func is not None:
            return self.ic_func(x)
        return torch.zeros_like(x)

    def get_f_target(self, x, t, L_):
        if self.source_func is not None:
            return self.source_func(x, t)
        return torch.zeros_like(x)


class DecoupledAdaptiveLambdaController:
    """
    Decoupled Multi-Adaptive Lambda Controller based on gradient-norm balancing (Wang et al. 2021).
    Dynamically balances PDE, IC, boundary, continuity, and Kirchhoff flux coupling losses.
    """
    def __init__(self, init_cont=10.0, init_flux=1.0, init_pde=1.0, init_ic=1.0,
                 lambda_min=0.01, lambda_max=500.0, alpha=0.1, adapt_every=10):
        self.lambda_cont = float(init_cont)
        self.lambda_flux = float(init_flux)
        self.lambda_pde = float(init_pde)
        self.lambda_ic = float(init_ic)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.alpha = float(alpha)
        self.adapt_every = int(adapt_every)

    def update(self, it, g_cont, g_flux, g_pde=None):
        if it % self.adapt_every != 0 or it < 10:
            return self.lambda_cont, self.lambda_flux

        norm_cont = float(torch.sqrt(torch.sum(g_cont ** 2)))
        norm_flux = float(torch.sqrt(torch.sum(g_flux ** 2)))

        ratio = float(np.clip(norm_cont / max(norm_flux, 1e-12), self.lambda_min, self.lambda_max))
        self.lambda_flux = float(np.clip((1.0 - self.alpha) * self.lambda_flux + self.alpha * ratio,
                                         self.lambda_min, self.lambda_max))
        return self.lambda_cont, self.lambda_flux


def _compute_l1_matrix_vectorized(alpha, grid_np):
    n = len(grid_np)
    if alpha <= 0:
        return torch.zeros(n, n, dtype=MATRIX_DTYPE).to(device)
    g1 = gamma(2 - alpha)
    h = np.diff(grid_np)
    I, J = np.tril_indices(n, k=-1)
    tau_ij = grid_np[I] - grid_np[J]
    tau_ij1 = grid_np[I] - grid_np[J + 1]
    w = (tau_ij ** (1 - alpha) - tau_ij1 ** (1 - alpha)) / (h[J] * g1)
    mat = np.zeros((n, n))
    np.add.at(mat, (I, J + 1), w)
    np.add.at(mat, (I, J), -w)
    return torch.tensor(mat, dtype=MATRIX_DTYPE).to(device)


def _compute_l21sigma_matrix(alpha, grid_np, sigma=None):
    n = len(grid_np)
    W = np.zeros((n, n))
    if sigma is None:
        sigma = 1.0 - alpha / 2.0
    tau = np.diff(grid_np)
    t_shift = grid_np[:-1] + sigma * tau
    g1 = gamma(1.0 - alpha)
    for i in range(n - 1):
        row = i + 1
        tn_sig = t_shift[i]
        for k in range(i + 1):
            term1 = (tn_sig - grid_np[k]) ** (1.0 - alpha)
            term2 = (tn_sig - grid_np[k + 1]) ** (1.0 - alpha) if tn_sig > grid_np[k + 1] else 0.0
            A_nk = (term1 - term2) / (1.0 - alpha)
            if k == 0:
                w = A_nk / g1
                W[row, 1] += w / tau[0]
                W[row, 0] -= w / tau[0]
            else:
                part1 = (2.0 * tn_sig - grid_np[k] - grid_np[k + 1]) * A_nk
                term3 = (tn_sig - grid_np[k]) ** (2.0 - alpha)
                term4 = (tn_sig - grid_np[k + 1]) ** (2.0 - alpha) if tn_sig > grid_np[k + 1] else 0.0
                part2 = 2.0 / (2.0 - alpha) * (term3 - term4)
                B_nk = part1 - part2
                c1 = (A_nk + B_nk / (tau[k] + tau[k - 1])) / g1
                c2 = -B_nk / (tau[k] + tau[k - 1]) / g1
                W[row, k + 1] += c1 / tau[k]
                W[row, k] -= c1 / tau[k]
                W[row, k] += c2 / tau[k - 1]
                W[row, k - 1] -= c2 / tau[k - 1]
    return torch.tensor(W, dtype=MATRIX_DTYPE).to(device)


def _compute_l21sigma_aux_weights(alpha, aux_grid, sigma):
    aux_grid = aux_grid.to(MATRIX_DTYPE)
    n_targets, M1 = aux_grid.shape
    M = M1 - 1
    assert M >= 2, 'L2-1sigma aux scheme needs aux_pts >= 2.'
    tau = aux_grid[:, 1:] - aux_grid[:, :-1]
    g1 = math.exp(math.lgamma(1.0 - alpha))
    eps = torch.tensor(1e-14, dtype=MATRIX_DTYPE, device=aux_grid.device)
    tn_sig = (aux_grid[:, -2] + sigma * tau[:, -1]).unsqueeze(-1)
    t_left = aux_grid[:, :-1]
    t_right = aux_grid[:, 1:]
    tau_j = torch.clamp(tn_sig - t_left, min=eps)
    tau_j1_raw = tn_sig - t_right
    mask = (tau_j1_raw > 0).to(MATRIX_DTYPE)
    tau_j1 = torch.clamp(tau_j1_raw, min=eps)
    A = (tau_j.pow(1.0 - alpha) - mask * tau_j1.pow(1.0 - alpha)) / (1.0 - alpha)
    weights = torch.zeros(n_targets, M1, dtype=MATRIX_DTYPE, device=aux_grid.device)
    w0 = A[:, 0] / g1
    weights[:, 1] += w0 / tau[:, 0]
    weights[:, 0] -= w0 / tau[:, 0]
    Ak = A[:, 1:]
    tk, tk1 = t_left[:, 1:], t_right[:, 1:]
    part1 = (2.0 * tn_sig - tk - tk1) * Ak
    term3 = tau_j[:, 1:].pow(2.0 - alpha)
    term4 = mask[:, 1:] * tau_j1[:, 1:].pow(2.0 - alpha)
    part2 = (2.0 / (2.0 - alpha)) * (term3 - term4)
    B = part1 - part2
    tau_k = tau[:, 1:]
    tau_km1 = tau[:, :-1]
    denom = tau_k + tau_km1
    c1 = (Ak + B / denom) / g1
    c2 = -B / denom / g1
    idx_k = torch.arange(1, M, device=aux_grid.device)
    weights[:, idx_k + 1] += c1 / tau_k
    weights[:, idx_k] -= c1 / tau_k
    weights[:, idx_k] += c2 / tau_km1
    weights[:, idx_k - 1] -= c2 / tau_km1
    return weights


def _build_aux_mesh(order, targets_t, M, r):
    sigma_paper = order / 2.0
    shift_denom = (1.0 - sigma_paper) + sigma_paper * ((M - 1.0) / M) ** r
    t_M = targets_t / shift_denom
    k = torch.linspace(0.0, 1.0, M + 1, dtype=MATRIX_DTYPE, device=device)
    aux_grid = t_M.view(-1, 1) * k.view(1, -1).pow(r)
    sigma_code = 1.0 - sigma_paper
    aux_weights = _compute_l21sigma_aux_weights(order, aux_grid, sigma_code)
    return aux_grid.to(NET_DTYPE), aux_weights.to(NET_DTYPE)


def _compute_l1_matrix_torch(alpha, grid_t):
    grid_t = grid_t.to(MATRIX_DTYPE)
    n = grid_t.shape[0]
    if n <= 1:
        return torch.zeros(n, n, dtype=MATRIX_DTYPE, device=grid_t.device)
    h = grid_t[1:] - grid_t[:-1]
    rows, cols = torch.tril_indices(n, n, offset=-1, device=grid_t.device)
    tau_ij = grid_t[rows] - grid_t[cols]
    tau_ij1 = grid_t[rows] - grid_t[cols + 1]
    eps = torch.tensor(1e-14, dtype=MATRIX_DTYPE, device=grid_t.device)
    p = 1.0 - alpha
    g1 = torch.exp(torch.lgamma(2.0 - alpha))
    w = (torch.clamp(tau_ij, min=eps) ** p - torch.clamp(tau_ij1, min=eps) ** p) / (torch.clamp(h[cols], min=eps) * g1)
    mat = torch.zeros((n, n), dtype=MATRIX_DTYPE, device=grid_t.device)
    mat.index_put_((rows, cols + 1), w, accumulate=True)
    mat.index_put_((rows, cols), -w, accumulate=True)
    return mat


class _PINNSolverBase:
    def __init__(self, graph, physics):
        self.graph = graph
        if isinstance(physics, list):
            self._physics_list = physics
            self.physics = physics[0]
        else:
            self._physics_list = None
            self.physics = physics
        self.compiled = False
        self.lambda_bc = 1.0
        self.models = {}
        self.current_epoch = 0
        self.hidden_layers = 4
        self.hidden_dim = 128
        self.use_fourier = False
        self.fourier_dim = 128
        self.fourier_sigma = 4.0
        self.fourier_sampling = 'gaussian'
        self.pts_per_unit = 250
        self.grading_factor = 1.5
        self.anchor_pts = 0
        self.constraint_mode = 'soft'
        self.bc_types = {}
        self.bc_values = {}
        self.adaptive_rate = 0.1
        self.adaptive_ema_beta = 0.95
        self.adaptive_lambda_min = 0.05
        self.adaptive_lambda_max = 1000.0
        self.adaptive_every = 25
        self.adaptive_warmup = 0
        self.adaptive_strategy_configs = {}
        self.dual_warmup_frac = 0.02
        self.dual_bdmm_end_frac = 0.10
        self.dual_gradient_end_frac = 0.95
        self.dual_phase_reset_ema = True
        self._frozen_lambda = None
        self.lr = 0.0005
        self.scheduler_min_lr = 1e-06
        self.validation_func = None
        self.inverse_enabled = False
        self.inverse_parameter_names = []
        self.inverse_param_bounds = {}
        self.inverse_include_alpha = False
        self.inverse_alpha_bounds = (1.05, 1.95)
        self.lambda_data = 1.0
        self.inverse_data = {}
        self.last_data_loss = 0.0
        self.use_singularity_capture = False
        self.xi_raw = nn.Parameter(torch.tensor(0.0, dtype=NET_DTYPE))
        self.xi_loss_adaptive = True
        self.xi_loss_ref = 0.1
        self.xi_loss_floor = 1e-05
        self.xi_lr_scale = 0.1
        self.xi_lr_scale_phase2 = 0.01

        self.dense_constraint_enabled = False
        self.dense_constraint_factor = 4
        self.dense_constraint_only_custom_flux = True

    def set_architecture(self, hidden_layers=4, hidden_dim=128, use_fourier=False, fourier_dim=128, fourier_sigma=4.0,
                         fourier_sampling='gaussian'):
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.use_fourier = use_fourier
        self.fourier_dim = fourier_dim
        self.fourier_sigma = fourier_sigma
        self.fourier_sampling = fourier_sampling
        return self

    def set_constraints(self, constraint_mode='soft', bc_types=None, bc_values=None):
        self.constraint_mode = constraint_mode.lower()
        self.bc_types = bc_types or {}
        self.bc_values = bc_values or {}
        return self

    def set_validation(self, func, **kwargs):
        self.validation_func = func
        return self

    def set_singularity_capture(self, enabled=True, xi_init=None, xi_loss_adaptive=True, xi_loss_ref=0.1,
                                xi_loss_floor=1e-05):
        self.use_singularity_capture = enabled
        self.xi_loss_adaptive = xi_loss_adaptive
        self.xi_loss_ref = float(xi_loss_ref)
        self.xi_loss_floor = float(xi_loss_floor)
        if enabled:
            init = float(xi_init) if xi_init is not None else 1.0
            self._xi_init_val = init
            self.xi_raw = nn.Parameter(torch.tensor(np.log(init), dtype=NET_DTYPE))
        return self

    def set_lr(self, lr=0.0005, min_lr=1e-06, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01):
        self.lr = lr
        self.scheduler_min_lr = min_lr
        self.xi_lr_scale = xi_lr_scale
        self.xi_lr_scale_phase2 = xi_lr_scale_phase2
        return self

    def set_dense_constraint_enforcement(self, enabled=True, factor=4, only_custom_flux=True):
       
        self.dense_constraint_enabled = bool(enabled)
        self.dense_constraint_factor = max(int(factor), 1)
        self.dense_constraint_only_custom_flux = bool(only_custom_flux)
        return self

    def _has_custom_flux(self):
        return any(hasattr(m.get('physics'), 'get_flux') for m in self.models.values())

    def _constraint_time_grid(self):
        if not getattr(self, 'dense_constraint_enabled', False):
            return self.t_grid
        if getattr(self, 'dense_constraint_only_custom_flux', True) and not self._has_custom_flux():
            return self.t_grid
        factor = max(int(getattr(self, 'dense_constraint_factor', 1)), 1)
        if factor <= 1:
            return self.t_grid

        t = self.t_grid.detach().cpu().numpy().reshape(-1)
        parts = [t[0]]
        for a, b in zip(t[:-1], t[1:]):
            for q in range(1, factor + 1):
                parts.append(a + (b - a) * (q / factor))
        t_ref = np.asarray(parts, dtype=float)
        return torch.tensor(t_ref, dtype=NET_DTYPE, device=device).view(-1, 1)

    def set_inverse_problem(self, parameter_names=None, include_alpha=False, param_bounds=None,
                            alpha_bounds=(1.05, 1.95), data_weight=1.0, initial_guesses=None):
        self.inverse_enabled = True
        self.inverse_parameter_names = list(parameter_names or [])
        self.inverse_param_bounds = dict(param_bounds or {})
        self.inverse_initial_guesses = dict(initial_guesses or {})
        self.inverse_include_alpha = bool(include_alpha)
        self.inverse_alpha_bounds = tuple(alpha_bounds)
        self.lambda_data = float(data_weight)
        return self

    @staticmethod
    def _to_bounded(raw, bounds):
        if bounds is None:
            return raw
        lo, hi = (float(bounds[0]), float(bounds[1]))
        return lo + (hi - lo) * torch.sigmoid(raw)

    @staticmethod
    def _raw_from_value(value, bounds):
        if bounds is None:
            return torch.tensor(float(value), dtype=NET_DTYPE, device=device)
        lo, hi = (float(bounds[0]), float(bounds[1]))
        v = float(np.clip((value - lo) / (hi - lo), 1e-06, 1.0 - 1e-06))
        return torch.tensor(math.log(v / (1.0 - v)), dtype=NET_DTYPE, device=device)

    def _edge_param_value(self, edge_idx, name):
        m = self.models[edge_idx]
        if 'inv_params' not in m or name not in m['inv_params']:
            return None
        cfg = m['inv_params'][name]
        return self._to_bounded(cfg['raw'], cfg['bounds'])

    def _refresh_edge_physics_parameters(self, edge_idx):
        m = self.models[edge_idx]
        if 'inv_params' not in m:
            return
        for pname, cfg in m['inv_params'].items():
            setattr(m['physics'], pname, self._to_bounded(cfg['raw'], cfg['bounds']))

    def get_estimated_parameters(self):
        out = {}
        for i, m in self.models.items():
            if 'inv_params' not in m:
                continue
            out[i] = {}
            for name, cfg in m['inv_params'].items():
                val = float(self._to_bounded(cfg['raw'], cfg['bounds']).detach().cpu().item())
                out[i][name] = val
                setattr(m['physics'], name, val)
        return out

    def _all_params(self):
        p = [p for m in self.models.values() for p in m['net'].parameters()]
        if getattr(self, 'use_singularity_capture', False):
            p = p + [self.xi_raw]
        if self.inverse_enabled:
            seen_inv = set()
            for m in self.models.values():
                if 'inv_params' in m:
                    for cfg in m['inv_params'].values():
                        if cfg['raw'] not in seen_inv:
                            p.append(cfg['raw'])
                            seen_inv.add(cfg['raw'])
        return p

    def _reinit_weights(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        for m in self.models.values():
            for layer in m['net'].modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                    nn.init.zeros_(layer.bias)
        self.lambda_bc = 1.0
        self._frozen_lambda = None
        for attr in ('_lagrange_ema_grad', '_lagrange_ema_bdmm'):
            if hasattr(self, attr):
                delattr(self, attr)
        if getattr(self, 'use_singularity_capture', False):
            init_val = getattr(self, '_xi_init_val', 1.0)
            with torch.no_grad():
                self.xi_raw.fill_(np.log(init_val))
            if hasattr(self, '_xi_lr_base'):
                del self._xi_lr_base

    def _v2_grad_norms(self, lp, ln, params):
        g_pde = torch.autograd.grad(lp, params, retain_graph=True, allow_unused=True)
        g_bc = torch.autograd.grad(ln, params, retain_graph=True, allow_unused=True)
        gp2 = sum(float((g.detach() ** 2).sum().cpu()) for g in g_pde if g is not None)
        gb2 = sum(float((g.detach() ** 2).sum().cpu()) for g in g_bc if g is not None)
        return math.sqrt(gp2), math.sqrt(gb2)

    def _apply_target_lambda_update(self, lam_target, ema_key):
        rho = float(np.clip(getattr(self, 'adaptive_rate', 0.05), 1e-6, 1.0))
        lam_min = float(getattr(self, 'adaptive_lambda_min', 1e-3))
        lam_max = float(getattr(self, 'adaptive_lambda_max', 100.0))
        beta = float(getattr(self, 'adaptive_ema_beta', 0.90))

        lam_target = float(np.clip(lam_target, lam_min, lam_max))
        prev_target = getattr(self, ema_key, lam_target)
        smoothed_target = math.exp(beta * math.log(prev_target) + (1 - beta) * math.log(lam_target))
        setattr(self, ema_key, smoothed_target)

        log_lam = (1 - rho) * math.log(self.lambda_bc) + rho * math.log(smoothed_target)
        self.lambda_bc = float(np.clip(math.exp(log_lam), lam_min, lam_max))

    def _update_gradient_ratio_v2(self, lp, ln, params):
        gp, gb = self._v2_grad_norms(lp, ln, params)
        lam_target = gp / max(gb, 1e-12)
        self._apply_target_lambda_update(lam_target, ema_key='_lagrange_ema_grad')
        return gp, gb, self.lambda_bc

    def _update_bdmm_v2(self, lp, ln):
        lp_val = max(float(lp.detach().cpu()), 1e-12)
        ln_val = max(float(ln.detach().cpu()), 1e-12)
        lam_target = lp_val / ln_val
        self._apply_target_lambda_update(lam_target, ema_key='_lagrange_ema_bdmm')
        return ln_val, self.lambda_bc

    def _compose_loss(self, lp, ln):
        lam_junc = getattr(self, 'lambda_junc', 1.0)
        loss_junc = getattr(self, '_current_loss_junc', 0.0)
        loss = lp + self.lambda_bc * ln + (self.lambda_bc * lam_junc) * loss_junc
        if getattr(self, 'ld_total', None) is not None and self.inverse_enabled:
            loss = loss + getattr(self, 'lambda_data', 1.0) * self.ld_total
        if getattr(self, 'l_data_total', None) is not None:
            loss = loss + self.l_data_total
        return loss

    def _run_lbfgs(self, params):
        pre_w = [p.data.clone() for p in params]
        with torch.enable_grad():
            lp0, ln0 = self.compute_losses()[:2]
            pre_loss = self._compose_loss(lp0, ln0).item()
        lbfgs = optim.LBFGS(params, max_iter=200, history_size=50, line_search_fn='strong_wolfe',
                            tolerance_change=1e-9, tolerance_grad=1e-7)

        def closure():
            lbfgs.zero_grad()
            lp_l, ln_l = self.compute_losses()[:2]
            loss_l = self._compose_loss(lp_l, ln_l)
            loss_l.backward()
            return loss_l

        lbfgs.step(closure)
        with torch.enable_grad():
            lp1, ln1 = self.compute_losses()[:2]
            post_loss = self._compose_loss(lp1, ln1).item()
        if post_loss > pre_loss * 2.0:
            for p, w in zip(params, pre_w):
                p.data.copy_(w)

    def _sequential_dual_phase(self, le, epochs):
        phase1 = int(getattr(self, 'dual_warmup_frac', 0.02) * epochs)
        phase2 = int(getattr(self, 'dual_bdmm_end_frac', 0.10) * epochs)
        phase3 = int(getattr(self, 'dual_gradient_end_frac', 0.95) * epochs)

        if le < phase1:
            return 'fixed', phase1, phase2, phase3
        if le < phase2:
            return 'bdmm_v2', phase1, phase2, phase3
        if le < phase3:
            return 'gradient_ratio_v2', phase1, phase2, phase3
        return 'fixed', phase1, phase2, phase3

    def _train_loop(self, epochs, strategy, use_lbfgs, optimizer, scheduler, epoch_offset=0, global_best=None):
        if not hasattr(self, 'lambda_bc'):
            self.lambda_bc = 1.0
        if not hasattr(self, '_lam_hist'): self._lam_hist = []
        if not hasattr(self, '_loss_hist'): self._loss_hist = []
        params = self._all_params()
        use_val = self.validation_func is not None
        best_l, best_w, best_lam = (float('inf'), None, self.lambda_bc)
        total = epoch_offset + epochs
        use_xi = getattr(self, 'use_singularity_capture', False)
        for le in range(epochs + 1):
            ep = epoch_offset + le
            self.current_epoch = ep
            if strategy in ('dual', 'dual_v2', 'multi_stage'):
                active, phase1, phase2, phase3 = self._sequential_dual_phase(le, epochs)
            else:
                active = strategy
                phase1 = phase2 = phase3 = None

            if strategy in ('dual', 'dual_v2', 'multi_stage') and phase3 is not None:
                if le == phase3 and self._frozen_lambda is None:
                    self._frozen_lambda = float(self.lambda_bc)
                    print(f'  [Dual freeze] λ = {self._frozen_lambda:.4f}')
                if le >= phase3 and self._frozen_lambda is not None:
                    self.lambda_bc = self._frozen_lambda
            if use_xi and strategy in ('dual', 'dual_v2', 'multi_stage') and (ep == total // 2):
                xi_lr_p2 = self.lr * getattr(self, 'xi_lr_scale_phase2', 0.01)
                for g in optimizer.param_groups:
                    if any((p is self.xi_raw for p in g['params'])):
                        g['lr'] = xi_lr_p2
                        break
            optimizer.zero_grad()
            lp, ln = self.compute_losses(ep)[:2]

            active_cfg = getattr(self, 'adaptive_strategy_configs', {}).get(active)
            if active_cfg:
                if 'adapt_every' in active_cfg:
                    self.adaptive_every = int(active_cfg['adapt_every'])
                if 'adaptive_rate' in active_cfg:
                    self.adaptive_rate = float(active_cfg['adaptive_rate'])
                if 'adaptive_ema_beta' in active_cfg:
                    self.adaptive_ema_beta = float(active_cfg['adaptive_ema_beta'])

            if active != getattr(self, '_last_adaptive_phase', None):
                if getattr(self, 'dual_phase_reset_ema', True):
                    for key in ('_lagrange_ema_grad', '_lagrange_ema_bdmm'):
                        if hasattr(self, key):
                            delattr(self, key)
                self._last_adaptive_phase = active

            adapt_every = max(int(getattr(self, 'adaptive_every', 10)), 1)
            adapt_warmup = int(getattr(self, 'adaptive_warmup', 0))
            if ep >= adapt_warmup and ep > 0 and ep % adapt_every == 0:
                lam_junc = getattr(self, 'lambda_junc', 1.0)
                loss_junc = getattr(self, '_current_loss_junc', 0.0)
                ln_for_adapt = ln + lam_junc * loss_junc
                if active == 'bdmm_v2':
                    self._update_bdmm_v2(lp, ln_for_adapt)
                elif active == 'gradient_ratio_v2':
                    self._update_gradient_ratio_v2(lp, ln_for_adapt, params)
                self._lam_hist.append((ep, self.lambda_bc))
            loss = self._compose_loss(lp, ln)
            if ep > 0 and ep % 10 == 0:
                self._loss_hist.append((ep, loss.item()))
            metric = self._eval_l2() if use_val else loss.item()
            if metric < best_l:
                best_l = metric
                best_w = [p.data.clone() for p in params]
                best_lam = self.lambda_bc
            if global_best is not None and metric < global_best['loss']:
                global_best.update(loss=metric, params=[p.data.clone() for p in params], lam=self.lambda_bc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            if use_xi and getattr(self, 'xi_loss_adaptive', True) and hasattr(self, '_xi_lr_base'):
                ref = getattr(self, 'xi_loss_ref', 0.1)
                floor = getattr(self, 'xi_loss_floor', 1e-05)
                scale = float(np.clip(lp.item() / ref, floor / ref, 1.0))
                for g in optimizer.param_groups:
                    if any((p is self.xi_raw for p in g['params'])):
                        g['lr'] = self._xi_lr_base * scale
                        break
            optimizer.step()
            scheduler.step()
            if le % 500 == 0:
                xi_s = f'  xi={torch.exp(self.xi_raw).item():.4f}' if use_xi else ''
                data_s = f' | Data={self.last_data_loss:.3e}' if self.inverse_enabled and self.inverse_data else ''
                junc_now = getattr(self, '_current_loss_junc', 0.0)
                junc_s = f' | JUNC={float(junc_now.detach().cpu() if torch.is_tensor(junc_now) else junc_now):.3e}' if self._has_custom_flux() else ''
                print(
                    f'Epoch {le:6d} | loss={loss.item():.3e} | PDE={lp.item():.3e} | BC={ln.item():.3e}{junc_s} | λ={self.lambda_bc:.3f}{xi_s}{data_s}')
        if best_w is not None:
            for p, w in zip(params, best_w): p.data.copy_(w)
            self.lambda_bc = best_lam
        if use_lbfgs:
            if getattr(self, '_frozen_lambda', None) is not None:
                self.lambda_bc = float(self._frozen_lambda)
                print(f'  [L-BFGS Handoff] Frozen λ = {self.lambda_bc:.4f}')
            else:
                recent = [lam for e, lam in self._lam_hist if e >= total - 1000]
                if recent:
                    self.lambda_bc = float(np.median(recent))
                    print(f'  [L-BFGS Handoff] Frozen λ = {self.lambda_bc:.4f}')
            self._run_lbfgs(params)
        if global_best is not None:
            if use_val:
                m2 = self._eval_l2()
            else:
                lp2, ln2 = self.compute_losses()[:2]
                m2 = self._compose_loss(lp2, ln2).item()
            if m2 < global_best['loss']:
                global_best.update(loss=m2, params=[p.data.clone() for p in params], lam=self.lambda_bc)
        return best_l

    def train(self, epochs=4000, strategy='dual', use_lbfgs=True):
        self._planned_epochs = epochs
        self._frozen_lambda = None
        self._last_adaptive_phase = None
        if not self.compiled:
            self.compile()
        params = self._all_params()
        opt = self._make_optimizer(params)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs + 1, eta_min=self.scheduler_min_lr)
        return self._train_loop(epochs, strategy, use_lbfgs, opt, sch)

    def train_multistart(self, epochs=4000, strategy='dual', use_lbfgs=True, n_starts=3, seed_base=0, probe_epochs=500,
                         lam_cfg=None):
        if not self.compiled:
            self.compile()

        if strategy != 'fixed':
            cfg = lam_cfg or dict(
                adapt_every=self.adaptive_every,
                adaptive_rate=self.adaptive_rate,
                adaptive_ema_beta=self.adaptive_ema_beta,
            )
            if strategy == 'dual_v2':
                if 'gradient_ratio_v2' not in cfg or 'bdmm_v2' not in cfg:
                    raise ValueError(
                        "dual_v2 requires lam_cfg with 'bdmm_v2' and 'gradient_ratio_v2' configs.")
                self.adaptive_strategy_configs = {
                    'bdmm_v2': dict(cfg['bdmm_v2']),
                    'gradient_ratio_v2': dict(cfg['gradient_ratio_v2']),
                }
                fallback = self.adaptive_strategy_configs['gradient_ratio_v2']
            else:
                self.adaptive_strategy_configs = {strategy: cfg}
                fallback = cfg
            self.set_adaptive_lambda(
                lambda_min=self.adaptive_lambda_min,
                lambda_max=self.adaptive_lambda_max,
                adapt_every=fallback.get('adapt_every', self.adaptive_every),
                adapt_rate=fallback.get('adaptive_rate', self.adaptive_rate),
                ema_beta=fallback.get('adaptive_ema_beta', self.adaptive_ema_beta),
                dual_warmup_frac=self.dual_warmup_frac,
                dual_bdmm_end_frac=self.dual_bdmm_end_frac,
                dual_gradient_end_frac=self.dual_gradient_end_frac,
            )


        params = self._all_params()
        global_best = dict(loss=float('inf'), params=None, lam=1.0)
        bp_score, bp_params, bp_lam = (float('inf'), None, 1.0)

        for s in range(n_starts):
            seed = seed_base * 100 + s * 37
            self._reinit_weights(seed=seed)
            opt = self._make_optimizer(params)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs + 1, eta_min=self.scheduler_min_lr)
            self._train_loop(probe_epochs, strategy, False, opt, sch, epoch_offset=0, global_best=global_best)
            if self.validation_func:
                score = self._eval_l2()
            else:
                lp_s, ln_s = self.compute_losses()[:2]
                score = self._compose_loss(lp_s, ln_s).item()
            lbl = 'L2' if self.validation_func else 'loss'
            xi_s = f'  xi={torch.exp(self.xi_raw).item():.4f}' if getattr(self, 'use_singularity_capture',
                                                                          False) else ''
            print(f'  probe {s + 1}/{n_starts}  seed={seed}  {lbl}={score:.3e}{xi_s}')
            if score < bp_score:
                bp_score = score
                bp_params = [p.data.clone() for p in params]
                bp_lam = self.lambda_bc

        if bp_params is not None:
            for p, w in zip(params, bp_params):
                p.data.copy_(w)
        self.lambda_bc = bp_lam
        self._frozen_lambda = None
        self._last_adaptive_phase = None
        print(f'  -> continuing best probe (score={bp_score:.3e}) for {epochs} epochs')

        opt = self._make_optimizer(params)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs + 1, eta_min=self.scheduler_min_lr)
        self._train_loop(epochs, strategy, use_lbfgs, opt, sch, epoch_offset=probe_epochs, global_best=global_best)

        if global_best['params'] is not None:
            for p, w in zip(params, global_best['params']):
                p.data.copy_(w)
            self.lambda_bc = global_best['lam']
            print(f"  -> global best L2 = {global_best['loss']:.3e}")

        return self

    def train_inverse(self, epochs=4000, strategy='dual', use_lbfgs=True):
        if not self.inverse_enabled:
            raise RuntimeError('Call set_inverse_problem(...) before train_inverse(...).')
        if not self.inverse_data:
            raise RuntimeError('Call generate_noisy_edge_data(...) before train_inverse(...).')
        return self.train(epochs=epochs, strategy=strategy, use_lbfgs=use_lbfgs)

    def set_adaptive_lambda(self, lambda_min=0.05, lambda_max=1000.0, adapt_every=25, adapt_rate=0.1, ema_beta=0.95,
                            dual_warmup_frac=0.02, dual_bdmm_end_frac=0.10, dual_gradient_end_frac=0.95):
        self.adaptive_lambda_min = lambda_min
        self.adaptive_lambda_max = lambda_max
        self.adaptive_every = adapt_every
        self.adaptive_rate = adapt_rate
        self.adaptive_ema_beta = ema_beta
        self.dual_warmup_frac = dual_warmup_frac
        self.dual_bdmm_end_frac = dual_bdmm_end_frac
        self.dual_gradient_end_frac = dual_gradient_end_frac
        return self

    def get_lambda_history(self):
        hist = getattr(self, '_lam_hist', [])
        return [{'epoch': e, 'lambda_after': lam} for e, lam in hist]

    def calibrate_lambda_controllers(self, window_epochs, seed=0, n_calib_epochs=150, target_reduction=0.05,
                                     settle_frac=0.25, beta_share=0.60, max_overhead_frac=0.02):
        if window_epochs <= 0:
            raise ValueError('window_epochs must be positive.')
        if not self.compiled:
            self.compile()
        probe = copy.deepcopy(self)
        probe._reinit_weights(seed=seed)
        probe.lambda_bc = float(self.lambda_bc)
        probe._planned_epochs = window_epochs
        params = probe._all_params()
        opt = probe._make_optimizer(params)
        probe_horizon = max(int(window_epochs), int(n_calib_epochs))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_horizon + 1, eta_min=probe.scheduler_min_lr)

        log_grad, log_bdmm = [], []
        for _ in range(int(n_calib_epochs)):
            opt.zero_grad()
            lp, ln = probe.compute_losses()[:2]
            gp, gb = probe._v2_grad_norms(lp, ln, params)
            log_grad.append(math.log(max(gp / max(gb, 1e-12), 1e-12)))
            lp_v = max(float(lp.detach().cpu()), 1e-12)
            ln_v = max(float(ln.detach().cpu()), 1e-12)
            log_bdmm.append(math.log(max(lp_v / ln_v, 1e-12)))
            loss = probe._compose_loss(lp, ln)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            scheduler.step()

        def beta_from_noise(log_series):
            if len(log_series) > 2:
                increments = np.diff(log_series)
                sigma = float(np.std(increments, ddof=1))
            else:
                sigma = 0.0
            r = float(np.clip(target_reduction, 1e-4, 0.99))
            beta = float(np.clip((1.0 - r) / (1.0 + r), 0.0, 0.999))
            return beta, sigma

        def config_for(extra_backward_cost, beta):
            adapt_every = max(1, round(float(extra_backward_cost) / max(max_overhead_frac, 1e-6)))
            tau_lambda = max((1.0 - beta_share) * settle_frac * window_epochs, 1.0)
            adaptive_rate = float(np.clip(adapt_every / tau_lambda, 0.01, 0.30))
            return {
                'adapt_every': int(adapt_every),
                'adaptive_rate': adaptive_rate,
                'adaptive_ema_beta': float(beta),
            }

        beta_grad, sigma_grad = beta_from_noise(log_grad)
        beta_bdmm, sigma_bdmm = beta_from_noise(log_bdmm)
        cfg = {
            'gradient_ratio_v2': config_for(extra_backward_cost=2.0, beta=beta_grad),
            'bdmm_v2': config_for(extra_backward_cost=0.0, beta=beta_bdmm),
        }
        print(f'[calibration|window={window_epochs}] grad_ratio: sigma_logdiff={sigma_grad:.3f} -> {cfg["gradient_ratio_v2"]}')
        print(f'[calibration|window={window_epochs}] bdmm:       sigma_logdiff={sigma_bdmm:.3f} -> {cfg["bdmm_v2"]}')
        del probe, opt, scheduler
        return cfg

    def run_strategy(self, epochs, strategy, initial_lambda, lam_cfg=None, use_lbfgs=True):
        self.lambda_bc = float(initial_lambda)
        if strategy != 'fixed':
            cfg = lam_cfg or dict(
                adapt_every=self.adaptive_every,
                adaptive_rate=self.adaptive_rate,
                adaptive_ema_beta=self.adaptive_ema_beta,
            )
            if strategy == 'dual_v2':
                if 'gradient_ratio_v2' not in cfg or 'bdmm_v2' not in cfg:
                    raise ValueError(
                        "dual_v2 requires lam_cfg with 'bdmm_v2' and 'gradient_ratio_v2' configs.")
                self.adaptive_strategy_configs = {
                    'bdmm_v2': dict(cfg['bdmm_v2']),
                    'gradient_ratio_v2': dict(cfg['gradient_ratio_v2']),
                }
                fallback = self.adaptive_strategy_configs['gradient_ratio_v2']
            else:
                self.adaptive_strategy_configs = {strategy: cfg}
                fallback = cfg
            self.set_adaptive_lambda(
                lambda_min=self.adaptive_lambda_min,
                lambda_max=self.adaptive_lambda_max,
                adapt_every=fallback.get('adapt_every', self.adaptive_every),
                adapt_rate=fallback.get('adaptive_rate', self.adaptive_rate),
                ema_beta=fallback.get('adaptive_ema_beta', self.adaptive_ema_beta),
                dual_warmup_frac=self.dual_warmup_frac,
                dual_bdmm_end_frac=self.dual_bdmm_end_frac,
                dual_gradient_end_frac=self.dual_gradient_end_frac,
            )
        self.train(epochs=epochs, strategy=strategy, use_lbfgs=use_lbfgs)
        rows = self.get_lambda_history()
        result = self.report_l2(self.validation_func)
        errors = result[0] if isinstance(result, tuple) else result
        global_rel = float(getattr(self, 'last_global_error', float('nan')))
        mean_rel = float(np.mean(errors)) if len(errors) else float('nan')
        return rows, global_rel, mean_rel

    def _rel_l2_on_grid(self, i, m, vf, x_np, x_t, dx):
        u_p = self._predict_np(i, x_t)
        u_e = self._exact_np(vf, x_np, m['L'])
        nrm_sq = np.sum(u_e ** 2) * dx
        err_sq = np.sum((u_p - u_e) ** 2) * dx
        norm_e = np.sqrt(nrm_sq)
        rel = np.sqrt(err_sq) / norm_e if norm_e >= 1e-10 else float('nan')
        return rel, err_sq, nrm_sq, norm_e

    def report_l2(self, exact_func, n_pts=200):
        print('\n--- L2 Error Report ---')
        errors_rand, errors_colloc = [], []
        total_sq_err_rand = total_sq_nrm_rand = 0.0
        total_sq_err_colloc = total_sq_nrm_colloc = 0.0
        with torch.no_grad():
            for i, m in self.models.items():
                vf = exact_func[i] if isinstance(exact_func, list) else exact_func
                _rng = np.random.default_rng(seed=i)
                x_rand_np = _rng.uniform(0, m['L'], n_pts)
                x_rand_t = torch.tensor(x_rand_np, dtype=NET_DTYPE).view(-1, 1).to(device)
                dx_rand = m['L'] / n_pts
                rel_r, err_sq_r, nrm_sq_r, norm_e_r = self._rel_l2_on_grid(i, m, vf, x_rand_np, x_rand_t, dx_rand)
                total_sq_err_rand += err_sq_r
                total_sq_nrm_rand += nrm_sq_r
                x_colloc_t = m['x_nsig_grid']
                x_colloc_np = x_colloc_t.detach().cpu().numpy().flatten()
                n_c = len(x_colloc_np)
                dx_colloc = m['L'] / max(1, n_c - 1)
                rel_c, err_sq_c, nrm_sq_c, norm_e_c = self._rel_l2_on_grid(i, m, vf, x_colloc_np, x_colloc_t, dx_colloc)
                total_sq_err_colloc += err_sq_c
                total_sq_nrm_colloc += nrm_sq_c
                if norm_e_r < 1e-10 and norm_e_c < 1e-10:
                    print(f"  Edge {i} {m['nodes']}: exact norm ~0, skipped")
                    continue
                if norm_e_r >= 1e-10:
                    errors_rand.append(rel_r)
                if norm_e_c >= 1e-10:
                    errors_colloc.append(rel_c)
                print(f"  Edge {i} {m['nodes']}: Rel L2 [Random]={rel_r:.4e} ({n_pts} pts)  "
                      f"| Rel L2 [Collocation]={rel_c:.4e} ({n_c} pts)")
        if errors_rand and total_sq_nrm_rand > 1e-10:
            self.last_global_error_random = float(np.sqrt(total_sq_err_rand / total_sq_nrm_rand))
            print(f'  Global Rel L2 [Random]:      {self.last_global_error_random:.4e}  '
                  f'(Mean: {np.mean(errors_rand):.4e})')
        if errors_colloc and total_sq_nrm_colloc > 1e-10:
            self.last_global_error_colloc = float(np.sqrt(total_sq_err_colloc / total_sq_nrm_colloc))
            print(f'  Global Rel L2 [Collocation]: {self.last_global_error_colloc:.4e}  '
                  f'(Mean: {np.mean(errors_colloc):.4e})')
        self.last_global_error = getattr(self, 'last_global_error_random',
                                         getattr(self, 'last_global_error_colloc', None))
        return errors_rand, errors_colloc

    def plot_results(self, exact_func=None, n_pts=200):
        n = len(self.models)
        fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n))
        if n == 1:
            axes = [axes]
        for i, m in self.models.items():
            x_np = np.linspace(0, m['L'], n_pts)
            x_t = torch.tensor(x_np, dtype=NET_DTYPE).view(-1, 1).to(device)
            u_p = self._predict_np(i, x_t)
            axes[i].plot(x_np, u_p, 'r-', label='PINN')
            if exact_func is not None:
                ef = exact_func[i] if isinstance(exact_func, list) else exact_func
                u_e = self._exact_np(ef, x_np, m['L'])
                axes[i].plot(x_np, u_e, 'k--', alpha=0.7, label='Exact')
            if getattr(self, 'use_anchors', False) and self.anchor_X and (i in self.anchor_X):
                axes[i].scatter(self.anchor_X[i].cpu().numpy().flatten(), self.anchor_U[i].cpu().numpy().flatten(), s=8,
                                c='blue', alpha=0.4, label='FD anchors')
            axes[i].set_title(f"Edge {i}  nodes={m['nodes']}")
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('u(x)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def post_process(self, u_exact_func=None):
        self.plot_results(exact_func=u_exact_func)

    def plot_graph_topology(self):
        G = nx.Graph()
        for u, v, L in self.graph.edges:
            G.add_edge(u, v, weight=L)
        plt.figure(figsize=(5, 3))
        nx.draw(G, with_labels=True, node_color='lightblue', font_weight='bold')
        plt.title('Metric Graph Topology')
        plt.show()

    def _make_optimizer(self, params):
        net_params = []
        inv_params = []
        for p in params:
            if getattr(self, 'use_singularity_capture', False) and p is self.xi_raw:
                continue
            is_inv = False
            if getattr(self, 'inverse_enabled', False):
                for m in self.models.values():
                    if 'inv_params' in m:
                        for cfg in m['inv_params'].values():
                            if p is cfg['raw']:
                                is_inv = True
            if is_inv:
                inv_params.append(p)
            else:
                net_params.append(p)
        param_groups = [{'params': net_params, 'lr': self.lr}]
        if inv_params:
            param_groups.append({'params': inv_params, 'lr': self.lr * 10.0})
        if getattr(self, 'use_singularity_capture', False):
            xi_lr = self.lr * getattr(self, 'xi_lr_scale', 0.1)
            self._xi_lr_base = xi_lr
            param_groups.append({'params': [self.xi_raw], 'lr': xi_lr})
        return optim.Adam(param_groups)

    def _predict_np(self, idx, x_t):
        raise NotImplementedError

    def _exact_np(self, vf, x_np, L):
        raise NotImplementedError

    def compile(self):
        raise NotImplementedError


    def predict(self, idx, *args):
        raise NotImplementedError

    def compute_losses(self, epoch=0):
        raise NotImplementedError

    def _eval_l2(self):
        raise NotImplementedError


class ParabolicPINNSolver(_PINNSolverBase):
    def __init__(self, graph, physics):
        super().__init__(graph, physics)
        self.mesh_type = 'power_law'
        self.pinn_pts = 200
        self.n_t = 100
        self.t_max = 1.0
        self.frac_scheme = 'L21sigma'
        self.l21_sigma = 0.5
        self.t_nsig_grid = None
        self.val_pts_per_unit = 500
        self.scheduler_min_lr = 1e-05
        self.kappa = 20.0
        self.aux_pts = 50
        self.aux_grading_factor = 2.0
        self.aux_t_grid = None
        self.aux_weights = None

    def set_mesh(self, mesh_type='power_law', pinn_pts=200, grading_factor=1.0, t_max=1.0, n_t=100):
        self.mesh_type = mesh_type
        self.pinn_pts = pinn_pts
        self.grading_factor = grading_factor
        self.t_max = t_max
        self.n_t = n_t
        return self

    def set_frac_scheme(self, scheme='L21sigma', sigma=None, aux_pts=None, aux_grading_factor=None):
        assert scheme in ('L1', 'L21sigma', 'L21_true', 'L1_aux', 'L21_aux'),\
            f"scheme must be one of 'L1', 'L21sigma', 'L21_true', 'L21_aux', got '{scheme}'"
        if scheme in ('L1_aux', 'L21_aux'):
            scheme = 'L21_aux'
        self.frac_scheme = scheme
        if scheme in ('L21sigma', 'L21_true'):
            alpha = getattr(self.physics, 'alpha', 0.5)
            self.l21_sigma = 1.0 - alpha / 2.0 if sigma is None else float(sigma)
        if scheme == 'L21_aux':
            if aux_pts is not None:
                self.aux_pts = int(aux_pts)
            if aux_grading_factor is not None:
                self.aux_grading_factor = float(aux_grading_factor)
        return self

    def set_validation(self, func, times=None, pts_per_unit=500):
        self.validation_func = func
        self.val_pts_per_unit = pts_per_unit
        return self

    def generate_noisy_edge_data(self, exact_func, n_points_per_edge=40, noise_std=0.01, seed=0):
        if not self.compiled:
            self.compile()
        rng = np.random.default_rng(seed)
        self.inverse_data = {}
        for i, m in self.models.items():
            n = max(int(n_points_per_edge), 4)
            x_np = rng.uniform(0.0, m['L'], size=n)
            t_np = rng.uniform(0.0, self.t_max, size=n)
            if isinstance(exact_func, list):
                clean = exact_func[i](x_np, t_np).reshape(-1)
            else:
                try:
                    clean = exact_func(x_np, t_np, i).reshape(-1)
                except TypeError:
                    clean = exact_func(x_np, t_np).reshape(-1)
            noisy = clean + rng.normal(0.0, float(noise_std), size=n)
            self.inverse_data[i] = dict(x=torch.tensor(x_np, dtype=NET_DTYPE, device=device).view(-1, 1),
                                        t=torch.tensor(t_np, dtype=NET_DTYPE, device=device).view(-1, 1),
                                        u_obs=torch.tensor(noisy, dtype=NET_DTYPE, device=device).view(-1, 1),
                                        noise_std=float(noise_std))
        return self.inverse_data

    def _generate_mesh(self, max_val, n_pts, is_spatial=False):
        if is_spatial:
            return np.linspace(0, max_val, n_pts)
        if self.mesh_type == 'power_law':
            idx = np.arange(n_pts)
            return max_val * (idx / (n_pts - 1)) ** self.grading_factor
        return np.linspace(0, max_val, n_pts)

    def compile(self):
        if self.use_singularity_capture:
            self.xi_raw = nn.Parameter(self.xi_raw.data.to(device))
        else:
            self.xi_raw = nn.Parameter(self.xi_raw.data.to(device))
        if self.frac_scheme == 'L21_aux':
            if self.constraint_mode == 'hard':
                raise NotImplementedError(
                    "frac_scheme='L21_aux' currently requires constraint_mode='soft' (the aux-mesh Caputo "
                    "quadrature has only been validated against the raw SDCNet output; combining it with the "
                    "'hard' exponential IC/BC blending in predict() has not been verified).")
        t_np = self._generate_mesh(self.t_max, self.n_t)
        self.t_grid = torch.tensor(t_np, dtype=NET_DTYPE).view(-1, 1).to(device)
        alpha = getattr(self.physics, 'alpha', None)
        if alpha is not None:
            if self.frac_scheme == 'L21sigma':
                if not hasattr(self, 'l21_sigma') or self.l21_sigma is None:
                    self.l21_sigma = 1.0 - alpha / 2.0
                self.Dt_alpha = _compute_l21sigma_matrix(alpha, t_np, self.l21_sigma).to(dtype=NET_DTYPE)
                t_nsig_np = np.empty(self.n_t)
                t_nsig_np[0] = t_np[0]
                t_nsig_np[1:] = (1.0 - self.l21_sigma) * t_np[:-1] + self.l21_sigma * t_np[1:]
                self.t_nsig_grid = torch.tensor(t_nsig_np, dtype=NET_DTYPE).view(-1, 1).to(device)
            elif self.frac_scheme == 'L21_aux':
                self.Dt_alpha = None
                self.t_nsig_grid = self.t_grid
                t_targets = self.t_grid[1:, 0].to(MATRIX_DTYPE)
                M = self.aux_pts
                r = self.aux_grading_factor
                self.aux_t_grid, self.aux_weights = _build_aux_mesh(alpha, t_targets, M, r)
                extra_evals = (self.n_t - 1) * (M + 1)
                baseline_evals = self.n_t
                print(f'  [mesh] scheme=L21_aux (uniform training grid, nonuniform L2-1sigma aux mesh per Eq. 3.6)  '
                      f'aux_pts={M}  aux_grading={r:.2f}  -> {extra_evals} network evals/edge/step for D^alpha u '
                      f'vs {baseline_evals} for the shared-mesh schemes (~{extra_evals / max(baseline_evals, 1):.1f}x)')
            else:
                self.Dt_alpha = _compute_l1_matrix_vectorized(alpha, t_np).to(dtype=NET_DTYPE)
                self.t_nsig_grid = self.t_grid
            if self.frac_scheme != 'L21_aux':
                tau_np = np.diff(t_np)
                print(
                    f'  [mesh] r={self.grading_factor:.2f}  scheme={self.frac_scheme}  tau_min={tau_np.min():.3e}  tau_max={tau_np.max():.3e}  ratio={tau_np.max() / tau_np.min():.1f}  Z(t)={self.use_singularity_capture}')
        net_in_dim = 2
        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = (edge[0], edge[1], float(edge[2]))
            n_x = max(int(self.pinn_pts * L), 2)
            x_np = self._generate_mesh(L, n_x, is_spatial=True)
            x_grid = torch.tensor(x_np, dtype=NET_DTYPE).view(-1, 1).to(device)
            if self.use_singularity_capture:
                net = SDCNet(smooth_dim=2, raw_dim=1, hidden_dim=self.hidden_dim,
                             hidden_layers=self.hidden_layers, use_fourier=self.use_fourier,
                             fourier_dim=self.fourier_dim, fourier_sigma=self.fourier_sigma,
                             fourier_sampling=self.fourier_sampling).to(device).to(NET_DTYPE)
            else:
                net = PINN_Net(in_dim=net_in_dim, hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers,
                               use_fourier=self.use_fourier, fourier_dim=self.fourier_dim,
                               fourier_sigma=self.fourier_sigma,
                               fourier_sampling=self.fourier_sampling).to(device).to(NET_DTYPE)
            T_mesh = self.t_grid.repeat_interleave(n_x, dim=0)
            T_mesh_nsig = self.t_nsig_grid.repeat_interleave(n_x, dim=0)
            phys_i = self._physics_list[i] if self._physics_list else self.physics
            f_target = phys_i.get_f_target(x_grid.repeat(self.n_t, 1), T_mesh_nsig, L).view(-1, 1).to(device)
            self.models[i] = dict(net=net, L=L, n_x=n_x, nodes=(u_node, v_node), x_grid=x_grid,
                                  X_mesh=x_grid.repeat(self.n_t, 1), T_mesh=T_mesh, T_mesh_nsig=T_mesh_nsig,
                                  f_target=f_target, physics=phys_i)
            if self.inverse_enabled:
                if self.inverse_include_alpha and self.frac_scheme != 'L1':
                    raise NotImplementedError("Inverse alpha estimation requires frac_scheme='L1'.")
                inv_params = {}
                for pname in self.inverse_parameter_names:
                    if not hasattr(phys_i, pname):
                        raise AttributeError(f"Physics on edge {i} has no attribute '{pname}'.")
                    bnd = self.inverse_param_bounds.get(pname, None)
                    inv_params[pname] = {
                        'raw': nn.Parameter(self._raw_from_value(float(getattr(phys_i, pname)), bnd).to(device)),
                        'bounds': bnd}
                if self.inverse_include_alpha:
                    ab = self.inverse_alpha_bounds
                    inv_params['alpha'] = {
                        'raw': nn.Parameter(self._raw_from_value(float(phys_i.alpha), ab).to(device)), 'bounds': ab}
                if inv_params:
                    self.models[i]['inv_params'] = inv_params
        self.compiled = True
        return self

    def predict(self, idx, x, t, return_components=False):
        m = self.models[idx]
        x_hat = (x / m['L']).to(NET_DTYPE)
        if self.use_singularity_capture:
            xi = torch.exp(self.xi_raw)
            Z = t.clamp(min=1e-30).pow(xi)
            x_smooth = torch.cat([x_hat, t], dim=-1)
            u_raw = m['net'](x_smooth, Z)
        else:
            inp = torch.cat([x_hat, t], dim=-1)
            u_raw = m['net'](inp)
        if self.constraint_mode == 'hard':
            _deg = {}
            for edge in self.graph.edges:
                _deg[edge[0]] = _deg.get(edge[0], 0) + 1
                _deg[edge[1]] = _deg.get(edge[1], 0) + 1
            u_node, v_node = m['nodes']
            tl = self.bc_types.get(u_node, 'junction' if _deg.get(u_node, 1) > 1 else 'dirichlet')
            tr = self.bc_types.get(v_node, 'junction' if _deg.get(v_node, 1) > 1 else 'dirichlet')
            vl = self._get_bc_value(u_node, t) if tl == 'dirichlet' else 0.0
            vr = self._get_bc_value(v_node, t) if tr == 'dirichlet' else 0.0
            L = m['L']
            if tl == 'dirichlet' and tr == 'dirichlet':
                dist_x = x * (L - x)
                lift = vl + (x / L) * (vr - vl)
            elif tl == 'dirichlet':
                dist_x = x
                lift = vl
            elif tr == 'dirichlet':
                dist_x = L - x
                lift = vr
            else:
                dist_x = 1.0
                lift = 0.0
            phys_i = m['physics']
            u_ic = phys_i.get_ic(x)
            kappa = getattr(self, 'kappa', 20.0)
            w = torch.exp(-kappa * t)
            u_final = w * u_ic + (1.0 - w) * (lift + dist_x * u_raw)
            if return_components:
                return u_final, u_final, None
            return u_final
        if return_components:
            return u_raw, u_raw, None
        return u_raw

    def _get_bc_value(self, node_idx, t):
        bc = self.bc_values.get(node_idx, 0.0)
        return bc(t) if callable(bc) else bc

    def _compute_aux_caputo_derivative(self, idx, m):
        n_x = m['n_x']
        n_targets = self.n_t - 1
        M1 = self.aux_pts + 1
        x_grid = m['x_grid']
        X_aux = x_grid.view(1, 1, n_x).expand(n_targets, M1, n_x).reshape(-1, 1)
        T_aux = self.aux_t_grid.unsqueeze(-1).expand(n_targets, M1, n_x).reshape(-1, 1)
        u_aux = self.predict(idx, X_aux, T_aux)
        u_aux = u_aux.view(n_targets, M1, n_x)
        dt_partial = torch.einsum('nm,nmx->nx', self.aux_weights, u_aux)
        zeros_row = torch.zeros(1, n_x, dtype=NET_DTYPE, device=device)
        return torch.cat([zeros_row, dt_partial], dim=0).reshape(-1, 1)

    def compute_losses(self, epoch=0):
        lp, ln = (0, 0)
        loss_junc = 0.0
        for i, m in self.models.items():
            if self.inverse_enabled and 'inv_params' in m:
                self._refresh_edge_physics_parameters(i)
            n_x_m = m['n_x']
            u_grid = self.predict(i, m['X_mesh'], m['T_mesh'])
            u_matrix_reshaped = u_grid.view(self.n_t, n_x_m)
            if self.inverse_enabled and 'inv_params' in m and ('alpha' in m['inv_params']):
                alpha_i = self._edge_param_value(i, 'alpha')
                Dt_i = _compute_l1_matrix_torch(alpha_i, self.t_grid.view(-1)).to(dtype=NET_DTYPE)
                dt_alpha_u = torch.mm(Dt_i, u_matrix_reshaped).view(-1, 1)
            elif self.frac_scheme == 'L21_aux':
                dt_alpha_u = self._compute_aux_caputo_derivative(i, m)
            else:
                dt_alpha_u = torch.mm(self.Dt_alpha, u_matrix_reshaped).view(-1, 1)
            X = m['X_mesh'].clone().detach().requires_grad_(True)
            T_eval = m['T_mesh_nsig'].clone().detach().requires_grad_(True)
            u_sp = self.predict(i, X, T_eval)
            u_x = torch.autograd.grad(u_sp, X, torch.ones_like(u_sp), create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0]
            f_target = m['f_target']
            res = m['physics'].F(X, T_eval, u_sp, u_x, u_xx, dt_alpha_u, f_target)
            lp += res.view(self.n_t, n_x_m)[1:].pow(2).mean()
        _deg = {}
        for edge in self.graph.edges:
            _deg[edge[0]] = _deg.get(edge[0], 0) + 1
            _deg[edge[1]] = _deg.get(edge[1], 0) + 1
        for i, m in self.models.items():
            if self.constraint_mode == 'soft':
                X_ic = m['x_grid']
                u_ic = self.predict(i, X_ic, torch.zeros_like(X_ic))
                ln += torch.mean((u_ic - m['physics'].get_ic(X_ic)) ** 2)
        for i, m in self.models.items():
            for node_idx, x_val in [(m['nodes'][0], 0.0), (m['nodes'][1], m['L'])]:
                if _deg.get(node_idx, 1) > 1:
                    continue
                bc_type = self.bc_types.get(node_idx, 'dirichlet')
                if self.constraint_mode == 'hard' and bc_type == 'dirichlet':
                    continue
                T_bc = self._constraint_time_grid().clone().detach().requires_grad_(True)
                X_bc = torch.full_like(T_bc, x_val).requires_grad_(True)
                u_bc = self.predict(i, X_bc, T_bc)
                if bc_type == 'dirichlet':
                    res_bc = (u_bc - self._get_bc_value(node_idx, T_bc)).pow(2).view(-1)
                    ln += res_bc.mean()
                else:
                    u_x_bc = torch.autograd.grad(u_bc, X_bc, torch.ones_like(u_bc), create_graph=True)[0]
                    res_nbc = (u_x_bc - self._get_bc_value(node_idx, T_bc)).pow(2).view(-1)
                    ln += res_nbc.mean()
        for node_idx, d in _deg.items():
            if d <= 1:
                continue
            inc = []
            for i, m in self.models.items():
                if node_idx == m['nodes'][0]:
                    inc.append((i, 0.0, 1.0))
                if node_idx == m['nodes'][1]:
                    inc.append((i, m['L'], -1.0))
            if len(inc) < 2:
                continue
            T_jc = self._constraint_time_grid().clone().detach()
            u_preds = []
            flux = torch.zeros_like(T_jc)
            uses_custom_flux = False
            for idx_e, xv, s in inc:
                X_j = torch.full_like(T_jc, xv).requires_grad_(True)
                T_j = T_jc.clone().requires_grad_(True)
                u_j = self.predict(idx_e, X_j, T_j)
                u_preds.append(u_j.view(-1))
                u_x_j = torch.autograd.grad(u_j, X_j, torch.ones_like(u_j), create_graph=True)[0]
                edge_physics = self.models[idx_e]['physics']
                if hasattr(edge_physics, 'get_flux'):
                    flux_val = edge_physics.get_flux(u_j, u_x_j)
                    flux = flux + flux_val * s
                    uses_custom_flux = True
                else:
                    flux = flux + u_x_j * s
            for a in range(len(u_preds)):
                for b in range(a + 1, len(u_preds)):
                    ct = (u_preds[a] - u_preds[b]).pow(2)
                    ln += ct.mean()
            bc_type = self.bc_types.get(node_idx, None)
            if bc_type == 'dirichlet':
                u_target = self._get_bc_value(node_idx, T_jc).view(-1)
                res_jc_bc = (u_preds[0] - u_target).pow(2)
                ln += res_jc_bc.mean()
            ft = flux.pow(2).view(-1)
            flux_loss_val = ft.mean()
            if uses_custom_flux:
                loss_junc += flux_loss_val
            else:
                ln = ln + flux_loss_val
        if ln == 0:
            ln = torch.tensor(0.0, requires_grad=True, dtype=NET_DTYPE).to(device)
        if self.inverse_enabled and self.inverse_data:
            ld = torch.tensor(0.0, dtype=NET_DTYPE, device=device)
            for i, obs in self.inverse_data.items():
                up = self.predict(i, obs['x'], obs['t'])
                ld = ld + torch.mean((up - obs['u_obs']) ** 2)
            self.ld_total = ld
            self.last_data_loss = float(ld.detach().cpu().item())
        else:
            self.ld_total = None
            self.last_data_loss = 0.0
        self._current_loss_junc = loss_junc
        return (lp, ln)

    def _eval_l2(self):
        if not hasattr(self, 'validation_func') or self.validation_func is None:
            return 0.0

        total_err_sq = 0.0
        total_ref_sq = 0.0

        for idx, edge in enumerate(self.graph.edges):
            L = edge[2]


            x_axis = np.linspace(0, L, getattr(self, 'pinn_pts', 100))
            t_axis = np.linspace(0, getattr(self, 't_max', 1.0), getattr(self, 'n_t', 200))


            X_mesh, T_mesh = np.meshgrid(x_axis, t_axis, indexing='ij')


            x_flat = X_mesh.ravel()
            t_flat = T_mesh.ravel()


            with torch.no_grad():
                x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=device).unsqueeze(1)
                t_tensor = torch.tensor(t_flat, dtype=torch.float32, device=device).unsqueeze(1)

                up = self.predict(idx, x_tensor, t_tensor).detach().cpu().numpy().ravel()


            exact_fn = self.validation_func[idx] if isinstance(self.validation_func, list) else self.validation_func
            ue = np.asarray(exact_fn(x_flat, t_flat)).ravel()


            err_sq = np.sum((up - ue) ** 2)
            ref_sq = np.sum(ue ** 2)

            total_err_sq += err_sq
            total_ref_sq += ref_sq


        rel_l2 = np.sqrt(total_err_sq) / (np.sqrt(total_ref_sq) + 1e-12)
        return float(rel_l2)

    def _predict_np(self, idx, x_t, t_val=None):
        t_val = self.t_max if t_val is None else float(t_val)
        t_t = torch.full_like(x_t, t_val)
        return self.predict(idx, x_t, t_t).detach().cpu().numpy().flatten()

    def _exact_np(self, vf, x_np, t_val):
        return np.asarray(vf(x_np, t_val)).flatten()

    def plot_results(self, exact_func=None, n_pts=200, plot_times=None):
        if plot_times is None:
            plot_times = [self.t_max * 0.25, self.t_max * 0.5, self.t_max * 0.75, self.t_max]
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(plot_times)))
        n = len(self.models)
        fig, axes = plt.subplots(n, 1, figsize=(9, 3.5 * n))
        if n == 1:
            axes = [axes]
        with torch.no_grad():
            for i, m in self.models.items():
                ax = axes[i]
                x_np = np.linspace(0, m['L'], n_pts)
                x_t = torch.tensor(x_np, dtype=NET_DTYPE).view(-1, 1).to(device)
                for k, t_val in enumerate(plot_times):
                    u_p = self._predict_np(i, x_t, t_val=t_val)
                    ax.plot(x_np, u_p, '-', color=colors[k], linewidth=2.0, label=f'PINN t={t_val:.2f}')
                    if exact_func is not None:
                        ef = exact_func[i] if isinstance(exact_func, list) else exact_func
                        u_e = self._exact_np(ef, x_np, t_val)
                        ax.plot(x_np, u_e, '--', color=colors[k], linewidth=1.2, alpha=0.7,
                                label=f'Exact t={t_val:.2f}')
                ax.set_title(f"Edge {i}  nodes={m['nodes']}")
                ax.set_xlabel('x')
                ax.set_ylabel('u(x, t)')
                ax.legend(fontsize=8, ncol=2)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def report_l2(self, exact_func):
        print('\n--- Parabolic L2 Error Report ---')
        errors = []
        total_sq_err = 0.0
        total_sq_nrm = 0.0
        with torch.no_grad():
            for i, m in self.models.items():
                vf = exact_func[i] if isinstance(exact_func, list) else exact_func
                n = max(int(getattr(self, 'val_pts_per_unit', 1000) * m['L']), 100)
                _rng = np.random.default_rng(seed=i)
                x_np = _rng.uniform(0, m['L'], n)
                t_np = _rng.uniform(0, self.t_max, n)
                x_t = torch.tensor(x_np, dtype=NET_DTYPE).view(-1, 1).to(device)
                t_t = torch.tensor(t_np, dtype=NET_DTYPE).view(-1, 1).to(device)
                up = self.predict(i, x_t, t_t).detach().cpu().numpy().flatten()
                ue = vf(x_np, t_np).flatten()
                vol = m['L'] * self.t_max
                dx_dt = vol / n
                nrm_sq = np.sum(ue ** 2) * dx_dt
                err_sq = np.sum((up - ue) ** 2) * dx_dt
                total_sq_nrm += nrm_sq
                total_sq_err += err_sq
                nrm = np.sqrt(nrm_sq)
                if nrm < 1e-10:
                    continue
                err = np.sqrt(err_sq) / nrm
                errors.append(err)
                print(f'  Edge {i}: {err:.4e}  ({n} pts)')
        if errors and total_sq_nrm > 1e-10:
            self.last_global_error = float(np.sqrt(total_sq_err / total_sq_nrm))
            print(f'  Global Rel L2: {self.last_global_error:.4e}  (Mean: {np.mean(errors):.4e})')
        return errors


class EllipticPINNSolver(_PINNSolverBase):
    def __init__(self, graph, physics):
        super().__init__(graph, physics)
        self.anchor_X = None
        self.anchor_U = None
        self.use_anchors = False
        self.lambda_data_schedule = None
        self.use_ntk_balance = False
        self.ntk_every = 200
        self.frac_scheme = 'L1'
        self.l21_sigma = None
        self.aux_pts = 50
        self.aux_grading_factor = 2.0

    def set_mesh(self, pts_per_unit=250, anchor_pts=0, grading_factor=1.5):
        self.pts_per_unit = pts_per_unit
        self.anchor_pts = anchor_pts
        self.grading_factor = grading_factor
        return self

    def set_frac_scheme(self, scheme='L1', sigma=None, aux_pts=None, aux_grading_factor=None):
        assert scheme in ('L1', 'L21sigma', 'L1_aux', 'L21_aux'),\
            f"scheme must be one of 'L1', 'L21sigma', 'L21_aux', got '{scheme}'"
        if scheme in ('L1_aux', 'L21_aux'):
            scheme = 'L21_aux'
        self.frac_scheme = scheme
        if scheme == 'L21sigma':
            alpha = getattr(self.physics, 'alpha', 1.5)
            self.l21_sigma = 1.0 - (alpha - 1.0) / 2.0 if sigma is None else float(sigma)
        if scheme == 'L21_aux':
            if aux_pts is not None:
                self.aux_pts = int(aux_pts)
            if aux_grading_factor is not None:
                self.aux_grading_factor = float(aux_grading_factor)
        return self

    def _compute_aux_fractional_term(self, idx, aux_grid, aux_weights, need_derivative):
        n_targets, M1 = aux_grid.shape
        X_aux = aux_grid.reshape(-1, 1)
        if need_derivative:
            X_aux = X_aux.clone().detach().requires_grad_(True)
            u_aux = self.predict(idx, X_aux)
            field = torch.autograd.grad(u_aux, X_aux, torch.ones_like(u_aux),
                                        create_graph=True, allow_unused=True)[0]
            if field is None:
                field = torch.zeros_like(u_aux)
        else:
            field = self.predict(idx, X_aux)
        field = field.view(n_targets, M1)
        d_partial = torch.sum(aux_weights * field, dim=1, keepdim=True)
        zero_row = torch.zeros(1, 1, dtype=NET_DTYPE, device=device)
        return torch.cat([zero_row, d_partial], dim=0)

    def set_lambda_data_schedule(self, schedule):
        self.lambda_data_schedule = schedule
        return self

    def set_ntk_balancing(self, enabled=True):
        self.use_ntk_balance = enabled
        return self

    def generate_noisy_edge_data(self, exact_func, n_points_per_edge=40, noise_std=0.01, seed=0):
        if not self.compiled:
            self.compile()
        rng = np.random.default_rng(seed)
        self.inverse_data = {}
        for i, m in self.models.items():
            n = max(int(n_points_per_edge), 4)
            x_np = rng.uniform(0.0, m['L'], size=n)
            if isinstance(exact_func, list):
                clean = exact_func[i](x_np).reshape(-1)
            else:
                try:
                    clean = exact_func(x_np, i).reshape(-1)
                except TypeError:
                    clean = exact_func(x_np).reshape(-1)
            noisy = clean + rng.normal(0.0, float(noise_std), size=n)
            self.inverse_data[i] = dict(x=torch.tensor(x_np, dtype=NET_DTYPE, device=device).view(-1, 1),
                                        u_obs=torch.tensor(noisy, dtype=NET_DTYPE, device=device).view(-1, 1),
                                        noise_std=float(noise_std))
        return self.inverse_data

    @staticmethod
    def _l1(alpha, grid_np):
        return _compute_l1_matrix_vectorized(alpha, grid_np)

    @staticmethod
    def _l1_torch(alpha, grid_t):
        return _compute_l1_matrix_torch(alpha, grid_t)

    @staticmethod
    def _l21sigma(alpha, grid_np, sigma):
        return _compute_l21sigma_matrix(alpha, grid_np, sigma)

    @staticmethod
    def compute_l1_matrix(alpha, x_np):
        return _compute_l1_matrix_vectorized(alpha, x_np)

    def _generate_mesh(self, L, n_pts):
        idx = np.arange(n_pts)
        return L * (idx / (n_pts - 1)) ** self.grading_factor

    def compile(self):
        alpha0 = self.physics.alpha
        assert 1.0 < alpha0 < 2.0, f'Elliptic engine requires 1 < alpha < 2, got alpha={alpha0}'
        self._deg = {}
        for edge in self.graph.edges:
            self._deg[edge[0]] = self._deg.get(edge[0], 0) + 1
            self._deg[edge[1]] = self._deg.get(edge[1], 0) + 1
        if self.use_singularity_capture:
            self.xi_raw = nn.Parameter(self.xi_raw.data.to(device))
        else:
            self.xi_raw = nn.Parameter(self.xi_raw.data.to(device))
        if self.frac_scheme == 'L21sigma' and self.l21_sigma is None:
            self.l21_sigma = 1.0 - (alpha0 - 1.0) / 2.0
        net_in_dim = 2 if self.use_singularity_capture else 1
        self.models = {}
        self.global_inv_params = None
        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = (edge[0], edge[1], float(edge[2]))
            phys_i = self._physics_list[i] if self._physics_list else self.physics
            n_pts = max(int(self.pts_per_unit * L), 2)
            aux_x_grid_a = aux_weights_a = aux_x_grid_b = aux_weights_b = None
            a = float(phys_i.alpha)
            if self.frac_scheme == 'L21_aux':
                x_np = self._generate_mesh(L, n_pts)
                x_grid = torch.tensor(x_np, dtype=NET_DTYPE).view(-1, 1).to(device)
                Da64 = Db64 = None
                x_nsig_np = x_np
                x_nsig_grid = x_grid
                targets_t = torch.tensor(x_np[1:], dtype=MATRIX_DTYPE, device=device)
                M, r = self.aux_pts, self.aux_grading_factor
                aux_x_grid_a, aux_weights_a = _build_aux_mesh(a - 1.0, targets_t, M, r)
                aux_x_grid_b, aux_weights_b = _build_aux_mesh(float(phys_i.beta), targets_t, M, r)
                extra_evals = (n_pts - 1) * (M + 1)

                print(f'  [elliptic mesh] edge {i}: scheme=L21_aux (graded training grid, nonuniform L2-1sigma '
                      f'aux mesh per Eq. 3.6)  aux_pts={M}  aux_grading={r:.2f}  -> {extra_evals} network evals '
                      f'for each of D^(a-1)[du/dx] and D^beta[u] vs {n_pts} for the shared-mesh schemes '
                      f'(~{extra_evals / max(n_pts, 1):.1f}x each)')
            else:
                x_np = self._generate_mesh(L, n_pts)
                x_grid = torch.tensor(x_np, dtype=NET_DTYPE).view(-1, 1).to(device)
                if self.frac_scheme == 'L21sigma':
                    Da64 = _compute_l21sigma_matrix(a - 1.0, x_np)
                    Db64 = _compute_l21sigma_matrix((phys_i.beta), x_np)
                    sig_a = 1.0 - (a - 1.0) / 2.0
                    x_nsig_np = np.empty(n_pts)
                    x_nsig_np[0] = x_np[0]
                    x_nsig_np[1:] = (1.0 - sig_a) * x_np[:-1] + sig_a * x_np[1:]
                    x_nsig_grid = torch.tensor(x_nsig_np, dtype=NET_DTYPE).view(-1, 1).to(device)
                else:
                    Da64 = self._l1(a - 1.0, x_np)
                    Db64 = self._l1(float(phys_i.beta), x_np)
                    x_nsig_np = x_np
                    x_nsig_grid = x_grid
            if self.use_singularity_capture:


                net = SDCNet(smooth_dim=1, raw_dim=1, hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers,
                             use_fourier=self.use_fourier, fourier_dim=self.fourier_dim,
                             fourier_sigma=self.fourier_sigma,
                             fourier_sampling=self.fourier_sampling).to(device).to(dtype=NET_DTYPE)
            else:
                net = PINN_Net(in_dim=net_in_dim, hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers,
                               use_fourier=self.use_fourier, fourier_dim=self.fourier_dim,
                               fourier_sigma=self.fourier_sigma,
                               fourier_sampling=self.fourier_sampling).to(device).to(dtype=NET_DTYPE)
            Da_fd_np = Da64.cpu().numpy() if Da64 is not None else np.zeros((n_pts, n_pts))
            Db_fd_np = Db64.cpu().numpy() if Db64 is not None else np.zeros((n_pts, n_pts))
            f_target = phys_i.get_f_target(x_np, x_nsig_np, L, Da_fd_np, Db_fd_np)
            Da = Da64.to(NET_DTYPE) if Da64 is not None else None
            Db = Db64.to(NET_DTYPE) if Db64 is not None else None
            self.models[i] = {'net': net, 'L': L, 'n': n_pts, 'nodes': (u_node, v_node), 'x_grid': x_grid, 'x_np': x_np,
                              'x_nsig_grid': x_nsig_grid, 'Da': Da, 'Db': Db,
                              'aux_x_grid_a': aux_x_grid_a, 'aux_weights_a': aux_weights_a,
                              'aux_x_grid_b': aux_x_grid_b, 'aux_weights_b': aux_weights_b,
                              'f_target': torch.tensor(f_target, dtype=NET_DTYPE).view(-1, 1).to(device),
                              'physics': phys_i}
            if self.inverse_enabled:
                if self.inverse_include_alpha and self.frac_scheme != 'L1':
                    raise NotImplementedError("Inverse alpha estimation requires frac_scheme='L1'.")
                if self.global_inv_params is None:
                    inv_params = {}
                    for pname in self.inverse_parameter_names:
                        bnd = self.inverse_param_bounds.get(pname, None)
                        init_val = self.inverse_initial_guesses.get(pname, float(getattr(phys_i, pname)))
                        inv_params[pname] = {
                            'raw': nn.Parameter(self._raw_from_value(init_val, bnd).to(device)),
                            'bounds': bnd}
                    if self.inverse_include_alpha:
                        ab = self.inverse_alpha_bounds
                        init_alpha = self.inverse_initial_guesses.get('alpha', float(phys_i.alpha))
                        inv_params['alpha'] = {
                            'raw': nn.Parameter(self._raw_from_value(init_alpha, ab).to(device)), 'bounds': ab}
                    self.global_inv_params = inv_params
                if self.global_inv_params:
                    self.models[i]['inv_params'] = self.global_inv_params
        if self.anchor_pts > 0:
            self._run_fd_inverse_solver()
            if self.lambda_data_schedule is None:
                total = getattr(self, '_planned_epochs', 4000)
                p1, p2 = (total // 4, total // 2)
                self.lambda_data_schedule = lambda ep, p1=p1, p2=p2: 0.1 if ep < p1 else 0.01 if ep < p2 else 0.0
        tau_np = np.diff(self._generate_mesh(1.0, max(int(self.pts_per_unit), 2)))
        print(
            f'  [elliptic mesh] scheme={self.frac_scheme}  n_pts/unit={self.pts_per_unit}  grading={self.grading_factor:.2f}  Z(x)={self.use_singularity_capture}')
        self.compiled = True
        return self

    def predict(self, idx, x):
        m = self.models[idx]
        if self.use_singularity_capture:
            xi = torch.exp(self.xi_raw)
            z = x.clamp(min=1e-30).pow(xi)
            u_raw = m['net'](x, z)
        else:
            u_raw = m['net'](x)
        if self.constraint_mode == 'hard':
            u_node, v_node = m['nodes']
            tl = self.bc_types.get(u_node, 'junction' if getattr(self, '_deg', {}).get(u_node, 1) > 1 else 'dirichlet')
            tr = self.bc_types.get(v_node, 'junction' if getattr(self, '_deg', {}).get(v_node, 1) > 1 else 'dirichlet')
            L = m['L']
            if tl == 'dirichlet' and tr == 'dirichlet':
                vl = self._get_bc_tensor(u_node)
                vr = self._get_bc_tensor(v_node)
                return vl + (x / L) * (vr - vl) + x * (L - x) * u_raw
            elif tl == 'dirichlet':
                vl = self._get_bc_tensor(u_node)
                return vl + x * u_raw
            elif tr == 'dirichlet':
                vr = self._get_bc_tensor(v_node)
                return vr + (L - x) * u_raw
            else:
                return u_raw
        return u_raw

    def _get_bc_tensor(self, node_idx):
        val = self.bc_values.get(node_idx, 0.0)
        return val() if callable(val) else torch.tensor(float(val), dtype=NET_DTYPE, device=device)

    def compute_losses(self, epoch=0):
        lp = torch.tensor(0.0, dtype=NET_DTYPE, device=device)
        ln = torch.tensor(0.0, dtype=NET_DTYPE, device=device)
        for i, m in self.models.items():
            if self.inverse_enabled and 'inv_params' in m:
                self._refresh_edge_physics_parameters(i)
            phys = m['physics']
            x_reg = m['x_grid'].clone().detach().requires_grad_(True)
            u_reg = self.predict(i, x_reg)
            du_reg = torch.autograd.grad(u_reg, x_reg, torch.ones_like(u_reg), create_graph=True, allow_unused=True)[0]
            if du_reg is None:
                du_reg = torch.zeros_like(u_reg)
            need_rebuild = self.inverse_enabled and 'inv_params' in m and (
                    'alpha' in m['inv_params'] or 'beta' in m['inv_params'])
            if need_rebuild:
                a_ord = self._edge_param_value(i, 'alpha') if 'alpha' in m['inv_params'] else torch.tensor(
                    float(phys.alpha), dtype=NET_DTYPE, device=device)
                b_ord = self._edge_param_value(i, 'beta') if 'beta' in m['inv_params'] else torch.tensor(
                    float(phys.beta), dtype=NET_DTYPE, device=device)
                Da_t = self._l1_torch(a_ord - 1.0, m['x_grid'].view(-1)).to(dtype=NET_DTYPE)
                Db_t = self._l1_torch(b_ord, m['x_grid'].view(-1)).to(dtype=NET_DTYPE)
                d_beta_u = torch.mm(Db_t, u_reg)
                d_alpha_u = torch.mm(Da_t, du_reg)
            elif self.frac_scheme == 'L21_aux':
                a_ord = torch.tensor(float(phys.alpha), dtype=NET_DTYPE, device=device)
                b_ord = torch.tensor(float(phys.beta), dtype=NET_DTYPE, device=device)
                d_alpha_u = self._compute_aux_fractional_term(
                    i, m['aux_x_grid_a'], m['aux_weights_a'], need_derivative=True)
                d_beta_u = self._compute_aux_fractional_term(
                    i, m['aux_x_grid_b'], m['aux_weights_b'], need_derivative=False)
            else:
                a_ord = torch.tensor(float(phys.alpha), dtype=NET_DTYPE, device=device)
                b_ord = torch.tensor(float(phys.beta), dtype=NET_DTYPE, device=device)
                d_beta_u = torch.mm(m['Db'], u_reg)
                d_alpha_u = torch.mm(m['Da'], du_reg)
            xt = m['x_nsig_grid'].clone().detach().requires_grad_(True)
            u_sp = self.predict(i, xt)
            du_sp = torch.autograd.grad(u_sp, xt, torch.ones_like(u_sp), create_graph=True, allow_unused=True)[0]
            if du_sp is None:
                du_sp = torch.zeros_like(u_sp)
            res = d_alpha_u - phys.F(xt, u_sp, du_sp, d_beta_u, m['f_target'])
            lp = lp + torch.mean(res.view(-1)[1:] ** 2)
        _deg = {}
        for edge in self.graph.edges:
            _deg[edge[0]] = _deg.get(edge[0], 0) + 1
            _deg[edge[1]] = _deg.get(edge[1], 0) + 1
        for node in self.graph.nodes:
            inc = []
            for i, m in self.models.items():
                if node == m['nodes'][0]:
                    inc.append((i, 0.0, 1.0))
                if node == m['nodes'][1]:
                    inc.append((i, m['L'], -1.0))
            if not inc:
                continue
            if len(inc) == 1:
                idx, xv, _ = inc[0]
                xi = torch.tensor([[xv]], dtype=NET_DTYPE, device=device, requires_grad=True)
                u_pred = self.predict(idx, xi)
                bc_type = self.bc_types.get(node, 'dirichlet')
                if self.constraint_mode == 'hard' and bc_type == 'dirichlet':
                    pass
                elif bc_type == 'dirichlet':
                    res_bc = (u_pred - self._get_bc_tensor(node)).pow(2).mean()
                    ln = ln + res_bc
                else:
                    du_p =\
                        torch.autograd.grad(u_pred, xi, torch.ones_like(u_pred), create_graph=True, allow_unused=True)[
                            0]
                    res_nbc = ((du_p if du_p is not None else torch.zeros_like(xi)) - self._get_bc_tensor(node)).pow(
                        2).mean()
                    ln = ln + res_nbc
            else:
                preds = []
                for idx_e, xv, _ in inc:
                    xi = torch.tensor([[xv]], dtype=NET_DTYPE, device=device)
                    preds.append(self.predict(idx_e, xi))
                for j in range(1, len(preds)):
                    ln = ln + (preds[0] - preds[j]).pow(2).mean()
                bc_type = self.bc_types.get(node, None)
                if bc_type == 'dirichlet':
                    res_bc = (preds[0] - self._get_bc_tensor(node)).pow(2).mean()
                    ln = ln + res_bc
                flux = torch.tensor(0.0, dtype=NET_DTYPE, device=device)
                for idx_e, xv, s in inc:
                    xi = torch.tensor([[xv]], dtype=NET_DTYPE, device=device, requires_grad=True)
                    u_val = self.predict(idx_e, xi)
                    grad = torch.autograd.grad(u_val, xi, torch.ones_like(u_val), create_graph=True, allow_unused=True)[
                        0]
                    if grad is not None:
                        flux = flux + grad * s
                ln = ln + flux.pow(2).mean()
        l_data = torch.tensor(0.0, dtype=NET_DTYPE, device=device)
        if self.use_anchors and self.anchor_X is not None:
            lam = self.lambda_data_schedule(epoch) if self.lambda_data_schedule is not None else 0.0
            if lam > 0.0:
                for i in self.anchor_X:
                    u_pred = self.predict(i, self.anchor_X[i])
                    l_data = l_data + lam * torch.mean((u_pred - self.anchor_U[i]) ** 2)
        if self.inverse_enabled and self.inverse_data:
            ld = torch.tensor(0.0, dtype=NET_DTYPE, device=device)
            for i, obs in self.inverse_data.items():
                up = self.predict(i, obs['x'])
                ld = ld + torch.mean((up - obs['u_obs']) ** 2)
            self.ld_total = ld
            self.last_data_loss = float(ld.detach().cpu().item())
        else:
            self.ld_total = None
            self.last_data_loss = 0.0
        if isinstance(ln, (int, float)) and ln == 0:
            ln = torch.tensor(0.0, dtype=NET_DTYPE, device=device, requires_grad=True)
        elif not getattr(ln, 'requires_grad', False):
            ln = ln.clone().detach().requires_grad_(True)
        return (lp, ln, l_data)

    def _eval_l2(self):
        if self.validation_func is None: return float('inf')
        total_sq_err = 0.0
        total_sq_nrm = 0.0
        with torch.no_grad():
            for i, m in self.models.items():
                vf = self.validation_func[i] if isinstance(self.validation_func, list) else self.validation_func
                x_t = torch.linspace(0, m['L'], 200, dtype=NET_DTYPE).view(-1, 1).to(device)
                u_p = self.predict(i, x_t).cpu().numpy().flatten()
                x_np = x_t.cpu().numpy().flatten()
                u_e = vf(x_np, m['L'])
                dx = m['L'] / max(1, len(x_np) - 1)
                nrm_sq = np.sum(u_e ** 2) * dx
                err_sq = np.sum((u_p - u_e) ** 2) * dx
                total_sq_nrm += nrm_sq
                total_sq_err += err_sq
        if total_sq_nrm > 1e-10:
            return float(np.sqrt(total_sq_err / total_sq_nrm))
        return float('inf')

    def _predict_np(self, idx, x_t):
        return self.predict(idx, x_t).detach().cpu().numpy().flatten()

    def _exact_np(self, vf, x_np, L):
        return np.asarray(vf(x_np, L)).flatten()

    def _run_fd_inverse_solver(self):
        alpha = self.physics.alpha
        self.anchor_X = {}
        self.anchor_U = {}
        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = (edge[0], edge[1], float(edge[2]))
            n_x = self.anchor_pts
            x_np = np.linspace(0, L, n_x)
            dx = x_np[1] - x_np[0]
            Da_fd = self._l1(alpha - 1.0, x_np).cpu().numpy()
            Db_fd = self._l1(self.physics.beta, x_np).cpu().numpy()
            f_np = self.physics.get_f_target(x_np, x_np, L, Da_fd, Db_fd)
            bc_left = float(self.bc_values.get(u_node, 0.0))
            bc_right = float(self.bc_values.get(v_node, 0.0))
            u = np.linspace(bc_left, bc_right, n_x)
            for _ in range(50):
                u_old = u.copy()
                du = np.zeros(n_x)
                du[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
                du[0] = (u[1] - u[0]) / dx
                du[-1] = (u[-1] - u[-2]) / dx
                n_int = n_x - 2
                res_int = np.array([(Da_fd @ du)[j + 1] - self.physics.F(x_np[j + 1], u[j + 1], du[j + 1],
                                                                         (Db_fd @ u)[j + 1], f_np[j + 1]) for j in
                                    range(n_int)])
                eps = 1e-06
                ab = np.zeros((3, n_int))
                ab[1] = np.array([((Da_fd @ np.gradient(
                    np.array([u[k] + (eps if k == j + 1 else 0) for k in range(n_x)]), dx))[j + 1] - self.physics.F(
                    x_np[j + 1], u[j + 1] + eps, du[j + 1], (Db_fd @ (u + eps * (np.arange(n_x) == j + 1)))[j + 1],
                    f_np[j + 1]) - res_int[j]) / eps for j in range(n_int)])
                for k in range(n_int):
                    j = k + 1
                    if k > 0:
                        u_m = u.copy()
                        u_m[j - 1] += eps
                        du_m = np.zeros(n_x)
                        du_m[1:-1] = (u_m[2:] - u_m[:-2]) / (2 * dx)
                        du_m[0] = (u_m[1] - u_m[0]) / dx
                        du_m[-1] = (u_m[-1] - u_m[-2]) / dx
                        r_m = (Da_fd @ du_m)[j] - self.physics.F(x_np[j], u_m[j], du_m[j], (Db_fd @ u_m)[j], f_np[j])
                        ab[2, k - 1] = (r_m - res_int[k]) / eps
                    if k < n_int - 1:
                        u_q = u.copy()
                        u_q[j + 1] += eps
                        du_q = np.zeros(n_x)
                        du_q[1:-1] = (u_q[2:] - u_q[:-2]) / (2 * dx)
                        du_q[0] = (u_q[1] - u_q[0]) / dx
                        du_q[-1] = (u_q[-1] - u_q[-2]) / dx
                        r_q = (Da_fd @ du_q)[j] - self.physics.F(x_np[j], u_q[j], du_q[j], (Db_fd @ u_q)[j], f_np[j])
                        ab[0, k + 1] = (r_q - res_int[k]) / eps
                delta = solve_banded((1, 1), ab, -res_int)
                u[1:-1] += delta
                u[0] = bc_left
                u[-1] = bc_right
                if np.max(np.abs(u - u_old)) < 1e-10:
                    break
            if not np.all(np.isfinite(u)):
                u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
            self.anchor_X[i] = torch.tensor(x_np, dtype=NET_DTYPE).view(-1, 1).to(device)
            self.anchor_U[i] = torch.tensor(u, dtype=NET_DTYPE).view(-1, 1).to(device)
        self.use_anchors = True


DEFAULT_ELLIPTIC_ARCH = dict(hidden_layers=4, hidden_dim=96, use_fourier=True, fourier_dim=64, fourier_sigma=8.0,
                             fourier_sampling='sobol')
DEFAULT_PARABOLIC_ARCH = dict(hidden_layers=4, hidden_dim=128, use_fourier=True, fourier_dim=64, fourier_sigma=1.0,
                              fourier_sampling='gaussian')


def run_elliptic_forward(graph, physics, exact_funcs, *, epochs=8000, n_starts=3, probe_epochs=600, seed_base=0,
                         pts_per_unit=250, grading_factor=1.5, arch=None, frac_scheme='L1', l21_sigma=None,
                         use_singularity_capture=False,
                         xi_init=None, lr=0.0005, min_lr=1e-06):
    arch = arch or DEFAULT_ELLIPTIC_ARCH
    solver = EllipticPINNSolver(graph, physics)
    solver.set_architecture(**arch)
    solver.set_mesh(pts_per_unit=pts_per_unit, anchor_pts=0, grading_factor=grading_factor)
    solver.set_constraints('soft')
    solver.set_frac_scheme(frac_scheme, sigma=l21_sigma)
    if use_singularity_capture:
        solver.set_singularity_capture(enabled=True, xi_init=xi_init)
    solver.set_lr(lr=lr, min_lr=min_lr)
    solver.set_validation(exact_funcs)
    solver.compile()
    solver.train_multistart(epochs=epochs, strategy='dual', use_lbfgs=True, n_starts=n_starts, seed_base=seed_base,
                            probe_epochs=probe_epochs)
    solver.report_l2(exact_funcs)
    solver.plot_results(exact_func=exact_funcs)
    return solver


def run_elliptic_inverse(graph, physics_list, exact_funcs_per_edge, *, parameter_names=('beta', 'reaction'),
                         include_alpha=False, param_bounds=None, alpha_bounds=(1.05, 1.95), data_weight=20.0,
                         n_points_per_edge=80, noise_std=0.01, seed=123, epochs=6000, pts_per_unit=220,
                         grading_factor=1.5, arch=None, frac_scheme='L1', l21_sigma=None,
                         use_singularity_capture=False, xi_init=None, lr=0.0005,
                         min_lr=1e-06):
    arch = arch or DEFAULT_ELLIPTIC_ARCH
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = EllipticPINNSolver(graph, physics_list)
    solver.set_architecture(**arch)
    solver.set_mesh(pts_per_unit=pts_per_unit, anchor_pts=0, grading_factor=grading_factor)
    solver.set_constraints('soft')
    solver.set_frac_scheme(frac_scheme, sigma=l21_sigma)
    if use_singularity_capture:
        solver.set_singularity_capture(enabled=True, xi_init=xi_init)
    solver.set_lr(lr=lr, min_lr=min_lr)
    solver.set_inverse_problem(parameter_names=list(parameter_names), include_alpha=include_alpha,
                               param_bounds=dict(param_bounds) if param_bounds else {}, alpha_bounds=alpha_bounds,
                               data_weight=data_weight)
    val_list = [lambda x, L, i=i: exact_funcs_per_edge[i](x) for i in range(len(physics_list))]
    solver.set_validation(val_list)
    solver.compile()
    solver.generate_noisy_edge_data(
        exact_func=[lambda x, i=i: exact_funcs_per_edge[i](x) for i in range(len(physics_list))],
        n_points_per_edge=n_points_per_edge, noise_std=noise_std, seed=seed)
    solver.train_inverse(epochs=epochs, strategy='dual', use_lbfgs=True)
    est = solver.get_estimated_parameters()
    solver.report_l2(val_list)
    return (solver, est)


def run_parabolic_forward_sweep(graph, physics_list, exact_u_per_edge, *, r_values=(1.0, 2.0, 4.0), scheme='L21sigma',
                                epochs=20000, n_starts=3, probe_epochs=1000, seed_base=0, pinn_pts=100, n_t=100,
                                t_max=1.0, arch=None, aux_pts=50):
    arch = arch or DEFAULT_PARABOLIC_ARCH
    n_edges = len(physics_list)
    results = {}
    for r in r_values:
        torch.manual_seed(seed_base)
        np.random.seed(seed_base)
        solver = ParabolicPINNSolver(graph, physics_list)
        if scheme in ('L1_aux', 'L21_aux'):
            solver.set_frac_scheme(scheme, aux_pts=aux_pts, aux_grading_factor=r)
        else:
            solver.set_frac_scheme(scheme)
        solver.set_architecture(**arch)
        solver.set_mesh(mesh_type='power_law', pinn_pts=pinn_pts, n_t=n_t, grading_factor=r, t_max=t_max)
        solver.set_constraints('soft')
        alpha0 = float(getattr(physics_list[0], 'alpha', 0.5))
        solver.set_singularity_capture(enabled=True, xi_init=alpha0, xi_loss_adaptive=True)
        solver.set_lr(lr=0.0005, min_lr=1e-05, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01)
        solver.set_validation([lambda x, t, i=i: exact_u_per_edge(i, x, t) for i in range(n_edges)], pts_per_unit=1000)
        solver.compile()
        solver.train_multistart(epochs=epochs, strategy='dual', use_lbfgs=True, n_starts=n_starts, seed_base=seed_base,
                                probe_epochs=probe_epochs)
        results[r] = solver.report_l2([lambda x, t, i=i: exact_u_per_edge(i, x, t) for i in range(n_edges)])
    return results


def parabolic_sweep_error_plot(results, r_values, alpha, scheme, n_edges, save_path='parabolic_star_sweep.png'):
    r_vals = [r for r in r_values if results.get(r)]
    if not r_vals:
        return
    means = [np.mean(results[r]) for r in r_vals]
    r_opt = (2.0 - alpha) / alpha if alpha > 0 else None
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'Parabolic star graph — per-edge mean rel L2\nalpha={alpha}, scheme={scheme}, {n_edges} edges',
                 fontsize=12)
    best_r = r_vals[int(np.argmin(means))]
    ax.semilogy(r_vals, means, 's-', color='steelblue', linewidth=2, markersize=8)
    ax.scatter([best_r], [min(means)], color='red', zorder=5, s=120, label=f'best r={best_r:.1f} ({min(means):.2e})')
    if r_opt is not None:
        ax.axvline(r_opt, color='red', linestyle='--', linewidth=1.2, label=f'opt r={r_opt:.1f}')
    ax.set_xlabel('Grading factor r')
    ax.set_ylabel('Mean rel L2')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xticks(list(r_values))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def run_parabolic_inverse(graph, physics_list, exact_u_per_edge, *, scheme='L1', r_grading=2.0, epochs=12000,
                          n_points_per_edge=60, noise_std=0.01, data_weight=20.0, seed=123, parameter_names=('nu',),
                          include_alpha=True, param_bounds=None, alpha_bounds=(0.2, 0.95), pinn_pts=100, n_t=100,
                          t_max=1.0, arch=None):
    arch = arch or DEFAULT_PARABOLIC_ARCH
    n_edges = len(physics_list)
    torch.manual_seed(seed)
    np.random.seed(seed)
    solver = ParabolicPINNSolver(graph, physics_list)
    solver.set_frac_scheme(scheme)
    solver.set_architecture(**arch)
    solver.set_mesh(mesh_type='power_law', pinn_pts=pinn_pts, n_t=n_t, grading_factor=r_grading, t_max=t_max)
    solver.set_constraints('soft')
    alpha0 = float(getattr(physics_list[0], 'alpha', 0.5))
    solver.set_singularity_capture(enabled=True, xi_init=alpha0, xi_loss_adaptive=True)
    solver.set_lr(lr=0.0005, min_lr=1e-05, xi_lr_scale=0.1, xi_lr_scale_phase2=0.01)
    pb = dict(param_bounds) if param_bounds else {}
    if 'nu' in parameter_names and 'nu' not in pb:
        pb['nu'] = (0.1, 2.0)
    solver.set_inverse_problem(parameter_names=list(parameter_names), include_alpha=include_alpha, param_bounds=pb,
                               alpha_bounds=alpha_bounds, data_weight=data_weight)
    exact_fns = [lambda x, t, i=i: exact_u_per_edge(i, x, t) for i in range(n_edges)]
    solver.compile()
    solver.generate_noisy_edge_data(exact_func=exact_fns, n_points_per_edge=n_points_per_edge, noise_std=noise_std,
                                    seed=seed)
    solver.train_inverse(epochs=epochs, strategy='dual', use_lbfgs=True)
    est = solver.get_estimated_parameters()
    solver.report_l2(exact_fns)
    return (solver, est)


class DeepONetPINNSolver(_PINNSolverBase):
    """
    Physics-Informed DeepONet Solver on Quantum / Metric Graphs.
    Uses pre-trained and frozen weights & biases from the adaptive DeepONet model
    (training_weights_biases/) as the starting point, and continues with the exact
    QGPINNS engine architecture:
      - Continuous space-time PDE evaluation via autograd
      - Boundary conditions, initial conditions, and vertex interface coupling
      - Kirchhoff nonlinear flux balance across internal junction vertices
      - Adaptive lambda balancing (dual_v2, gradient_ratio_v2, bdmm_v2, decoupled, fixed)
      - Two-phase training: Adam optimization followed by L-BFGS second-order polish!
    """
    def __init__(self, graph, physics, weights_dir='training_weights_biases', edge_types=None, branch_inputs=None, eps=0.1, t_max=1.0):
        super().__init__(graph, physics)
        self.weights_dir = weights_dir
        self.edge_types = dict(edge_types) if edge_types else {}
        self.branch_inputs = dict(branch_inputs) if branch_inputs else {}
        self.eps = float(eps)
        self.t_max = float(t_max)
        self.pinn_pts = 100
        self.n_t = 100
        self.decoupled_controller = None

    def set_mesh(self, pinn_pts=100, n_t=100, t_max=1.0):
        self.pinn_pts = pinn_pts
        self.n_t = n_t
        self.t_max = t_max
        return self

    def set_branch_inputs(self, branch_inputs):
        self.branch_inputs.update(branch_inputs)
        if self.compiled:
            for i, b in branch_inputs.items():
                if i in self.models:
                    self.models[i]['net'].set_branch_input(b)
        return self

    def _deduce_edge_type(self, edge_idx):
        if edge_idx in self.edge_types:
            return self.edge_types[edge_idx].lower()
        u_node, v_node, _ = self.graph.edges[edge_idx]
        deg = {}
        for e in self.graph.edges:
            deg[e[0]] = deg.get(e[0], 0) + 1
            deg[e[1]] = deg.get(e[1], 0) + 1
        if deg.get(u_node, 1) == 1:
            return 'inflow'
        if deg.get(v_node, 1) == 1:
            return 'outflow'
        return 'inner'

    def compile(self):
        t_np = np.linspace(0.0, self.t_max, self.n_t, dtype=np.float32)
        self.t_grid = torch.tensor(t_np, dtype=torch.float32, device=device).view(-1, 1)

        for i, edge in enumerate(self.graph.edges):
            u_node, v_node, L = edge[0], edge[1], float(edge[2])
            n_x = max(int(self.pinn_pts * L), 2)
            x_np = np.linspace(0.0, L, n_x, dtype=np.float32)
            x_grid = torch.tensor(x_np, dtype=torch.float32, device=device).view(-1, 1)

            m_type = self._deduce_edge_type(i)
            b_in = self.branch_inputs.get(i, None)

            # Initialize PretrainedDeepONet with the saved weights & biases as the starting point!
            net = PretrainedDeepONet(model_type=m_type, weights_dir=self.weights_dir, u_branch=b_in).to(device)

            X_mesh = x_grid.repeat(self.n_t, 1)
            T_mesh = self.t_grid.repeat_interleave(n_x, dim=0)

            phys_i = self._physics_list[i] if self._physics_list else self.physics
            f_target = phys_i.get_f_target(X_mesh, T_mesh, L).view(-1, 1).to(device)

            self.models[i] = dict(
                net=net,
                L=L,
                n_x=n_x,
                nodes=(u_node, v_node),
                x_grid=x_grid,
                X_mesh=X_mesh,
                T_mesh=T_mesh,
                f_target=f_target,
                physics=phys_i,
                edge_type=m_type
            )
        self.compiled = True
        return self

    def predict(self, idx, x, t):
        m = self.models[idx]
        L = m['L']
        x_norm = x / L
        return m['net'](x_norm, t)

    def compute_losses(self, epoch=0):
        lp = torch.tensor(0.0, device=device)
        ln = torch.tensor(0.0, device=device)
        loss_junc = torch.tensor(0.0, device=device)

        # 1. Edge PDE Residuals
        for i, m in self.models.items():
            L = m['L']
            X = m['X_mesh'].clone().detach().requires_grad_(True)
            T = m['T_mesh'].clone().detach().requires_grad_(True)
            X_norm = X / L
            TX = torch.cat([T, X_norm], dim=-1)

            u_sp = m['net'].forward_trunk(TX)
            B = m['net'].forward_branch()
            if B.ndim == 1:
                B = B.unsqueeze(0)
            if B.shape[0] != u_sp.shape[0]:
                B = B.expand(u_sp.shape[0], -1)
            u = torch.sum(B * u_sp, dim=-1, keepdim=True)

            d_tx = torch.autograd.grad(u, TX, torch.ones_like(u), create_graph=True)[0]
            d_t = d_tx[:, 0:1]
            d_x = d_tx[:, 1:2] / L
            d_xx = torch.autograd.grad(d_x, TX, torch.ones_like(d_x), create_graph=True)[0][:, 1:2] / L

            f_target = m['f_target']
            res = m['physics'].F(X, T, u, d_x, d_xx, d_t, f_target)
            lp = lp + res.pow(2).mean()

        # 2. Initial Condition Loss (t = 0)
        for i, m in self.models.items():
            if self.constraint_mode == 'soft':
                X_ic = m['x_grid']
                T_ic = torch.zeros_like(X_ic)
                u_ic = self.predict(i, X_ic, T_ic)
                ln = ln + torch.mean((u_ic - m['physics'].get_ic(X_ic)) ** 2)

        # 3. Boundary Condition Loss at Leaf/Boundary Nodes
        deg = {}
        for edge in self.graph.edges:
            deg[edge[0]] = deg.get(edge[0], 0) + 1
            deg[edge[1]] = deg.get(edge[1], 0) + 1

        for i, m in self.models.items():
            for node_idx, x_val in [(m['nodes'][0], 0.0), (m['nodes'][1], m['L'])]:
                if deg.get(node_idx, 1) > 1:
                    continue
                bc_type = self.bc_types.get(node_idx, 'dirichlet')
                T_bc = self.t_grid.clone().detach().requires_grad_(True)
                X_bc = torch.full_like(T_bc, x_val).requires_grad_(True)
                u_bc = self.predict(i, X_bc, T_bc)
                if bc_type == 'dirichlet':
                    res_bc = (u_bc - self._get_bc_value(node_idx, T_bc)).pow(2)
                    ln = ln + res_bc.mean()
                else:
                    u_x_bc = torch.autograd.grad(u_bc, X_bc, torch.ones_like(u_bc), create_graph=True)[0]
                    res_nbc = (u_x_bc - self._get_bc_value(node_idx, T_bc)).pow(2)
                    ln = ln + res_nbc.mean()

        # 4. Multi-Edge Junction Conditions (Continuity and Kirchhoff Flux Conservation)
        for node_idx, d in deg.items():
            if d <= 1:
                continue
            inc = []
            for i, m in self.models.items():
                if node_idx == m['nodes'][0]:
                    inc.append((i, 0.0, 1.0))
                if node_idx == m['nodes'][1]:
                    inc.append((i, m['L'], -1.0))
            if len(inc) < 2:
                continue

            T_jc = self.t_grid.clone().detach()
            u_preds = []
            flux = torch.zeros_like(T_jc)
            for idx_e, xv, s in inc:
                X_j = torch.full_like(T_jc, xv).requires_grad_(True)
                T_j = T_jc.clone().requires_grad_(True)
                u_j = self.predict(idx_e, X_j, T_j)
                u_preds.append(u_j.view(-1))
                u_x_j = torch.autograd.grad(u_j, X_j, torch.ones_like(u_j), create_graph=True)[0]
                edge_physics = self.models[idx_e]['physics']
                if hasattr(edge_physics, 'get_flux'):
                    flux_val = edge_physics.get_flux(u_j, u_x_j)
                else:
                    flux_val = -self.eps * u_x_j + 1.0 * u_j * (1.0 - u_j)
                flux = flux + flux_val * s

            for a in range(len(u_preds)):
                for b in range(a + 1, len(u_preds)):
                    ln = ln + (u_preds[a] - u_preds[b]).pow(2).mean()

            flux_loss = flux.pow(2).mean()
            loss_junc = loss_junc + flux_loss
            ln = ln + flux_loss

        self._current_loss_junc = loss_junc
        return (lp, ln)


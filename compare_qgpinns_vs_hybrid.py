"""
Comparison: Original QGPINNs vs Pretrained DeepONet Hybrid
==========================================================
Test Graph: Chain Graph G1 (7 edges, 8 vertices)
Governing PDE: Space-Time Drift-Diffusion dt(rho) - eps*dxx(rho) + nu*(1-2*rho)*dx(rho) = 0
Reference: High-Fidelity Finite Volume Method (FVM) Solver

Models evaluated:
  1. Original QGPINNs: Randomly initialized PINN_Net on each edge, trained with QGPINNs physics loss.
  2. DeepONet Hybrid: Pre-trained adaptive DeepONet weights as starting point, then continued with QGPINNs.

Generates:
  - Error comparison (Relative L2, Absolute L2, Linf)
  - Time comparison (Wall-clock training time & speedup)
  - Comprehensive comparison figure saved to results/qgpinns_vs_deeponet_hybrid_comparison.png
"""
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'Engine'))
sys.path.insert(0, os.path.join(BASE_DIR, 'scratch', 'official_repo'))

from QGPINNs_Engine import (
    DeepONetPINNSolver,
    PretrainedDeepONet,
    DriftDiffusionPhysics,
    DecoupledAdaptiveLambdaController,
    PINN_Net
)

device = torch.device('cpu')

from src.graph import Example5
from src.quantumGraphSolverFVM import QuantumGraphSolverFVM
from src.GPs import RBF, get_sample_fns
import jax
import jax.numpy as jnp
from jax import random


def main():
    print("=" * 85)
    print("  HEAD-TO-HEAD BENCHMARK: ORIGINAL QGPINNS vs DEEPONET HYBRID")
    print(f"  Target Hardware: {device}")
    print("  Test Graph: Chain Graph G1 (7 edges, 8 nodes)")
    print("=" * 85)

    os.makedirs('results', exist_ok=True)
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'training_weights_biases')

    # 1. Setup Graph & PDE Physics
    EPS = 0.1
    seed = 0
    graph_fvm = Example5(eps=EPS, n_edges=7)
    ne = graph_fvm.ne
    L = 1.0

    class MetricGraphTopology:
        nodes = list(range(graph_fvm.n_v))
        edges = [(int(e[0]), int(e[1]), L) for e in graph_fvm.E]

    metric_graph = MetricGraphTopology()

    edge_list = []
    inner_idx = 0
    for i, e in enumerate(graph_fvm.E):
        inflow_idx = np.where(e[0] == graph_fvm.inflowNodes)[0]
        outflow_idx = np.where(e[1] == graph_fvm.outflowNodes)[0]
        if len(inflow_idx) > 0:
            edge_list.append([i, inflow_idx[0], *e, -1])
        elif len(outflow_idx) > 0:
            edge_list.append([i, outflow_idx[0], *e, 1])
        else:
            edge_list.append([i, inner_idx, *e, 0])
            inner_idx += 1
    edge_list = np.vstack(edge_list)

    n_inflow_edges = np.sum(edge_list[:, -1] == -1)
    n_outflow_edges = np.sum(edge_list[:, -1] == 1)
    n_inner_edges = np.sum(edge_list[:, -1] == 0)

    # 2. Flow Parameterization Setup
    N_DATA_BC = 100
    N_DATA_INIT = 100
    x_sensor_bc = np.linspace(0, 1, N_DATA_BC + 1)
    x_sensor_init = np.linspace(0, 1, N_DATA_INIT + 1)
    n_sensor_bc = N_DATA_BC + 1

    n_rbf_flow = 10
    x_rbf_flow = np.linspace(0, 1, n_rbf_flow)
    K_flow_jnp = RBF(jnp.array(x_sensor_bc).reshape((n_sensor_bc, 1)),
                     jnp.array(x_rbf_flow).reshape((-1, 1)),
                     (1.0, 0.2))
    K_flow = np.array(K_flow_jnp)

    # 3. Initial & Boundary Conditions
    sample_u_in_fn, sample_u_out_fn, sample_u_init_fn = get_sample_fns(N=468, length_scale=0.4)
    global_key = random.key(seed)
    global_key, key = random.split(global_key, 2)
    key, inflow_key, outflow_key, init_key, vel_key, beta_key = random.split(key, 6)

    inflow_fn_list = [sample_u_in_fn(s) for s in random.split(inflow_key, len(graph_fvm.inflowNodes))]
    outflow_fn_list = [sample_u_out_fn(s) for s in random.split(outflow_key, len(graph_fvm.outflowNodes))]
    init_cond_fn_list = [sample_u_init_fn(s) for s in random.split(init_key, graph_fvm.ne)]

    def dirichletAlpha(x):
        alpha = np.zeros(graph_fvm.n_v)
        alpha[graph_fvm.inflowNodes] = np.array([f(x) for f in inflow_fn_list])
        return alpha

    def dirichletBeta(x):
        beta_out = np.zeros(graph_fvm.n_v)
        beta_out[graph_fvm.outflowNodes] = np.array([f(x) for f in outflow_fn_list])
        return beta_out

    graph_fvm.dirichletAlpha = dirichletAlpha
    graph_fvm.dirichletBeta = dirichletBeta
    graph_fvm.initial_cond = lambda x: np.array([f(x) for f in init_cond_fn_list])
    graph_fvm.initial_cond_jnp = lambda x: jnp.array([f(x) for f in init_cond_fn_list])

    vel = np.array(random.uniform(vel_key, (graph_fvm.ne,), minval=0.5, maxval=2.0))
    graph_fvm.v = vel
    print(f"Edge Velocities: {np.round(vel, 3)}")

    # 4. Reference High-Fidelity FVM Solution
    print("\nComputing Reference Ground Truth via High-Fidelity FVM Solver...")
    FVM_NT = 2000
    FVM_NX = 1000
    fnx = round(FVM_NX / 100)
    fnt = round(FVM_NT / 100)
    TimeOffset = 0.1
    graph_fvm.lb[0] = -TimeOffset
    fvm_solver = QuantumGraphSolverFVM(graph_fvm)
    v_raw = fvm_solver.solve(nx=FVM_NX + 1, nt=int(FVM_NT * (1 + TimeOffset)) + 1)
    v = np.array([fvm_solver.get_u_edge(v_raw.T, i).T for i in range(graph_fvm.ne)])
    v = v[:, :, int(TimeOffset * FVM_NT):]

    u_init_list = [np.interp(x_sensor_init, np.linspace(0, 1, FVM_NX + 1), v[i, :, 0]) for i in range(ne)]
    u_in = np.hstack([f(x_sensor_bc) for f in inflow_fn_list]).reshape(-1, len(x_sensor_bc))
    u_out = np.hstack([f(x_sensor_bc) for f in outflow_fn_list]).reshape(-1, len(x_sensor_bc))
    v_ref = v[:, ::fnx, ::fnt]  # (7, 101, 101)
    ref_norm = float(np.sqrt(np.mean(v_ref ** 2)))
    print(f"Reference FVM Solution Grid: {v_ref.shape} (L2 Norm: {ref_norm:.4f})")

    # Common evaluation grid (101 x 101)
    t_eval_np = np.linspace(0, 1, 101)
    x_eval_np = np.linspace(0, 1, 101)
    TX_eval = np.hstack([l.reshape((-1, 1)) for l in np.meshgrid(t_eval_np, x_eval_np)])  # (t, x)
    TX_eval_t = torch.tensor(TX_eval, dtype=torch.float32, device=device)

    # ---------------------------------------------------------------------------------
    # MODEL 1: ORIGINAL QGPINNs (Random Initialization from Scratch)
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  [MODEL 1] TRAINING ORIGINAL QGPINNs (FROM SCRATCH)")
    print("=" * 80)

    cache_qgp_path = os.path.join(BASE_DIR, 'results', 'model1_qgp_cache.npz')
    if os.path.exists(cache_qgp_path):
        print(f"Found cached Original QGPINNs results at {cache_qgp_path}! Loading cache...")
        cached = np.load(cache_qgp_path)
        pred_qgp = cached['pred_qgp']
        loss_history_qgp = list(cached['loss_history_qgp'])
        time_qgp = float(cached['time_qgp'])
        abs_err_qgp = float(cached['abs_err_qgp'])
        rel_err_qgp = float(cached['rel_err_qgp'])
        linf_qgp = float(cached['linf_qgp'])
        print(f"Loaded QGPINNs: Rel L2 Error = {rel_err_qgp * 100:.2f}% | Abs L2 = {abs_err_qgp:.4e} | Time = {time_qgp:.2f} s")
    else:
        torch.manual_seed(42)
        qgp_nets = nn.ModuleList([
            PINN_Net(in_dim=2, hidden_dim=128, hidden_layers=4, use_fourier=True, fourier_dim=64).to(device).to(torch.float32)
            for _ in range(ne)
        ])

        opt_qgp = torch.optim.Adam(qgp_nets.parameters(), lr=1e-3)
        ctrl_qgp = DecoupledAdaptiveLambdaController(init_cont=10.0, init_flux=1.0, alpha=0.1, adapt_every=10)

        # Collocation grid for QGPINNs (35 x 35 = 1225 points per edge)
        nx_colloc, nt_colloc = 35, 35
        x_c = torch.linspace(0, 1, nx_colloc, device=device)
        t_c = torch.linspace(0, 1, nt_colloc, device=device)
        T_mesh, X_mesh = torch.meshgrid(t_c, x_c, indexing='ij')
        TX_pde = torch.stack([T_mesh.flatten(), X_mesh.flatten()], dim=-1).requires_grad_(True)

        # Boundary and interface coordinates
        t_bc = torch.linspace(0, 1, nt_colloc, device=device).unsqueeze(1)
        tx_0 = torch.cat([t_bc, torch.zeros_like(t_bc)], dim=-1).requires_grad_(True)
        tx_1 = torch.cat([t_bc, torch.ones_like(t_bc)], dim=-1).requires_grad_(True)

        # Initial condition coordinates
        x_ic = torch.linspace(0, 1, nx_colloc, device=device).unsqueeze(1)
        tx_ic = torch.cat([torch.zeros_like(x_ic), x_ic], dim=-1)

        u_init_t = [torch.tensor(np.interp(np.linspace(0, 1, nx_colloc), x_sensor_init, u_init_list[i]), dtype=torch.float32, device=device).unsqueeze(1) for i in range(ne)]
        u_in_t = torch.tensor(np.interp(np.linspace(0, 1, nt_colloc), x_sensor_bc, u_in[0]), dtype=torch.float32, device=device).unsqueeze(1)
        u_out_t = torch.tensor(np.interp(np.linspace(0, 1, nt_colloc), x_sensor_bc, u_out[0]), dtype=torch.float32, device=device).unsqueeze(1)

        loss_history_qgp = []
        time_history_qgp = []
        start_qgp = time.time()

        epochs_adam_qgp = 1200
        for it in range(1, epochs_adam_qgp + 1):
            opt_qgp.zero_grad()
            loss_pde = torch.tensor(0.0, device=device)
            loss_ic = torch.tensor(0.0, device=device)

            u_at_0 = []
            u_at_1 = []
            flux_at_0 = []
            flux_at_1 = []

            for i in range(ne):
                net = qgp_nets[i]
                # PDE residual
                u_p = net(TX_pde)
                g = torch.autograd.grad(u_p, TX_pde, torch.ones_like(u_p), create_graph=True)[0]
                d_t = g[:, 0:1]
                d_x = g[:, 1:2]
                d_xx = torch.autograd.grad(d_x, TX_pde, torch.ones_like(d_x), create_graph=True)[0][:, 1:2]
                res = d_t - EPS * d_xx + vel[i] * (1.0 - 2.0 * u_p) * d_x
                loss_pde = loss_pde + res.pow(2).mean()

                # IC loss
                u_ic_pred = net(tx_ic)
                loss_ic = loss_ic + (u_ic_pred - u_init_t[i]).pow(2).mean()

                # Boundary predictions
                u0 = net(tx_0)
                gx0 = torch.autograd.grad(u0, tx_0, torch.ones_like(u0), create_graph=True)[0][:, 1:2]
                j0 = -EPS * gx0 + vel[i] * u0 * (1.0 - u0)
                u_at_0.append(u0)
                flux_at_0.append(j0)

                u1 = net(tx_1)
                gx1 = torch.autograd.grad(u1, tx_1, torch.ones_like(u1), create_graph=True)[0][:, 1:2]
                j1 = -EPS * gx1 + vel[i] * u1 * (1.0 - u1)
                u_at_1.append(u1)
                flux_at_1.append(j1)

            # Boundary losses
            loss_bc = (u_at_0[0] - u_in_t).pow(2).mean() + (u_at_1[-1] - u_out_t).pow(2).mean()

            # Continuity and Kirchhoff flux coupling across junctions
            loss_cont = torch.tensor(0.0, device=device)
            loss_flux = torch.tensor(0.0, device=device)
            for i in range(ne - 1):
                loss_cont = loss_cont + (u_at_1[i] - u_at_0[i + 1]).pow(2).mean()
                loss_flux = loss_flux + (flux_at_1[i] - flux_at_0[i + 1]).pow(2).mean()

            if it % ctrl_qgp.adapt_every == 0 and it >= 10:
                params_list = list(qgp_nets.parameters())
                gc = torch.autograd.grad(loss_cont, params_list, retain_graph=True)
                gf = torch.autograd.grad(loss_flux, params_list, retain_graph=True)
                norm_c = torch.sqrt(torch.stack([p.pow(2).sum() for p in gc]).sum())
                norm_f = torch.sqrt(torch.stack([p.pow(2).sum() for p in gf]).sum())
                ctrl_qgp.update(it, norm_c, norm_f)

            total_loss_qgp = loss_pde + 10.0 * loss_ic + 10.0 * loss_bc + ctrl_qgp.lambda_cont * loss_cont + ctrl_qgp.lambda_flux * loss_flux
            total_loss_qgp.backward()
            opt_qgp.step()

            loss_history_qgp.append(total_loss_qgp.item())
            time_history_qgp.append(time.time() - start_qgp)

            if it % 200 == 0 or it == epochs_adam_qgp:
                print(f"  Step {it:4d}/{epochs_adam_qgp} | Loss: {total_loss_qgp.item():.4e} | PDE: {loss_pde.item():.4e} | Cont: {loss_cont.item():.4e} | Flux: {loss_flux.item():.4e}")

        # L-BFGS polish for QGPINNs
        print("  Running L-BFGS polish for Original QGPINNs...")
        lbfgs_qgp = torch.optim.LBFGS(qgp_nets.parameters(), lr=0.1, max_iter=20, history_size=10)
        for _ in range(15):
            def closure_qgp():
                lbfgs_qgp.zero_grad()
                loss_eval = 0.0
                for i in range(ne):
                    u_p = qgp_nets[i](TX_pde)
                    g = torch.autograd.grad(u_p, TX_pde, torch.ones_like(u_p), create_graph=True)[0]
                    d_t, d_x = g[:, 0:1], g[:, 1:2]
                    d_xx = torch.autograd.grad(d_x, TX_pde, torch.ones_like(d_x), create_graph=True)[0][:, 1:2]
                    res = d_t - EPS * d_xx + vel[i] * (1.0 - 2.0 * u_p) * d_x
                    loss_eval = loss_eval + res.pow(2).mean()
                loss_eval.backward()
                return loss_eval
            try:
                lbfgs_qgp.step(closure_qgp)
            except Exception:
                pass

        time_qgp = time.time() - start_qgp
        print(f"Original QGPINNs Total Training Time: {time_qgp:.2f} s")

        # Evaluate Original QGPINNs Predictions
        pred_qgp = []
        with torch.no_grad():
            for i in range(ne):
                out = qgp_nets[i](TX_eval_t).cpu().numpy().reshape(101, 101)
                pred_qgp.append(out)
        pred_qgp = np.stack(pred_qgp, axis=0)
        abs_err_qgp = float(np.sqrt(np.mean((pred_qgp - v_ref) ** 2)))
        rel_err_qgp = abs_err_qgp / ref_norm
        linf_qgp = float(np.max(np.abs(pred_qgp - v_ref)))

        print(f"Original QGPINNs Rel L2 Error: {rel_err_qgp * 100:.2f}% | Abs L2: {abs_err_qgp:.4e} | Linf: {linf_qgp:.4e}")

        # Cache baseline results
        np.savez(cache_qgp_path,
                 pred_qgp=pred_qgp,
                 loss_history_qgp=np.array(loss_history_qgp),
                 time_qgp=time_qgp,
                 abs_err_qgp=abs_err_qgp,
                 rel_err_qgp=rel_err_qgp,
                 linf_qgp=linf_qgp)
        print(f"Cached Original QGPINNs baseline results to {cache_qgp_path}")

    # ---------------------------------------------------------------------------------
    # MODEL 2: DEEPONET HYBRID (Adaptive DeepONet Weights as Starting Point)
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  [MODEL 2] TRAINING DEEPONET HYBRID (WARM-START FROM SAVED WEIGHTS)")
    print("=" * 80)

    edge_types = {}
    for i, e in enumerate(edge_list):
        if e[-1] == -1:
            edge_types[i] = 'inflow'
        elif e[-1] == 0:
            edge_types[i] = 'inner'
        else:
            edge_types[i] = 'outflow'

    solver_hyb = DeepONetPINNSolver(
        graph=metric_graph,
        physics=[DriftDiffusionPhysics(edge_idx=i, eps=EPS, nu=vel[i]) for i in range(ne)],
        weights_dir=WEIGHTS_DIR,
        edge_types=edge_types,
        eps=EPS,
        t_max=1.0,
        device=device
    )
    solver_hyb.compile()

    beta_hyb = nn.Parameter(torch.zeros(n_inflow_edges + 2 * n_inner_edges + n_outflow_edges, n_rbf_flow, dtype=torch.float32, device=device))
    K_flow_t = torch.tensor(K_flow, dtype=torch.float32, device=device)

    def get_hybrid_u_base(beta):
        z = beta @ K_flow_t.T
        u_list = []
        in_offset = n_inflow_edges
        out_offset = 2 * n_inner_edges + n_inflow_edges
        for e in edge_list:
            v_val = torch.tensor([vel[e[0]]], dtype=torch.float32, device=device)
            init_val = torch.tensor(u_init_list[e[0]], dtype=torch.float32, device=device)
            if e[-1] == -1:
                u_in_t = torch.tensor(u_in[e[1]], dtype=torch.float32, device=device)
                u_list.append(torch.cat([u_in_t, z[e[1]], init_val, v_val]))
            elif e[-1] == 0:
                u_list.append(torch.cat([z[2 * e[1] + in_offset], z[2 * e[1] + in_offset + 1], init_val, v_val]))
            else:
                u_out_t = torch.tensor(u_out[e[1]], dtype=torch.float32, device=device)
                u_list.append(torch.cat([z[out_offset + e[1]], u_out_t, init_val, v_val]))
        return torch.stack(u_list)

    opt_hyb = torch.optim.AdamW([beta_hyb], lr=1e-4)
    ctrl_hyb = DecoupledAdaptiveLambdaController(init_cont=10.0, init_flux=1.0, alpha=0.1, adapt_every=10)

    tx_in_h = torch.tensor(np.hstack([np.linspace(0, 1, 101)[:, None], np.ones((101, 1))]), dtype=torch.float32, device=device)
    tx_out_h = torch.tensor(np.hstack([np.linspace(0, 1, 101)[:, None], np.zeros((101, 1))]), dtype=torch.float32, device=device)

    def compute_hyb_residuals(beta):
        u_b = get_hybrid_u_base(beta)
        val_0, flux_0, val_1, flux_1 = [], [], [], []
        for i, m in solver_hyb.models.items():
            net = m['net']
            u_e = u_b[i]
            t0 = tx_out_h.clone().requires_grad_(True)
            u0 = net(t0, u=u_e)
            gx0 = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0][:, 1:2]
            j0 = -EPS * gx0 + vel[i] * u0 * (1.0 - u0)
            val_0.append(u0)
            flux_0.append(j0)

            t1 = tx_in_h.clone().requires_grad_(True)
            u1 = net(t1, u=u_e)
            gx1 = torch.autograd.grad(u1, t1, torch.ones_like(u1), create_graph=True)[0][:, 1:2]
            j1 = -EPS * gx1 + vel[i] * u1 * (1.0 - u1)
            val_1.append(u1)
            flux_1.append(j1)

        loss_c = torch.tensor(0.0, device=device)
        loss_f = torch.tensor(0.0, device=device)
        for i in range(ne - 1):
            loss_c = loss_c + torch.sum((val_1[i] - val_0[i + 1]) ** 2)
            loss_f = loss_f + torch.sum((flux_1[i] - flux_0[i + 1]) ** 2)
        return loss_c / (ne - 1), loss_f / (ne - 1)

    loss_history_hyb = []
    time_history_hyb = []
    start_hyb = time.time()

    epochs_adam_hyb = 800
    for it in range(1, epochs_adam_hyb + 1):
        opt_hyb.zero_grad()
        lc, lf = compute_hyb_residuals(beta_hyb)
        if it % ctrl_hyb.adapt_every == 0 and it >= 10:
            gc = torch.autograd.grad(lc, beta_hyb, retain_graph=True)[0]
            gf = torch.autograd.grad(lf, beta_hyb, retain_graph=True)[0]
            ctrl_hyb.update(it, gc, gf)

        tot_hyb = ctrl_hyb.lambda_cont * lc + ctrl_hyb.lambda_flux * lf
        tot_hyb.backward()
        opt_hyb.step()

        loss_history_hyb.append(tot_hyb.item())
        time_history_hyb.append(time.time() - start_hyb)

        if it % 200 == 0 or it == epochs_adam_hyb:
            print(f"  Step {it:4d}/{epochs_adam_hyb} | Loss: {tot_hyb.item():.4e} | Cont: {lc.item():.4e} | Flux: {lf.item():.4e} | lam_cont: {ctrl_hyb.lambda_cont:.2f} | lam_flux: {ctrl_hyb.lambda_flux:.2f}")

    # L-BFGS polish for DeepONet Hybrid
    print("  Running L-BFGS second-order polish for DeepONet Hybrid...")
    lbfgs_hyb = torch.optim.LBFGS([beta_hyb], lr=0.5, max_iter=20, history_size=20)
    for _ in range(25):
        def closure_hyb():
            lbfgs_hyb.zero_grad()
            lc, lf = compute_hyb_residuals(beta_hyb)
            loss = ctrl_hyb.lambda_cont * lc + ctrl_hyb.lambda_flux * lf
            loss.backward()
            return loss
        lbfgs_hyb.step(closure_hyb)

    time_hyb = time.time() - start_hyb
    print(f"DeepONet Hybrid Total Training Time: {time_hyb:.2f} s")

    # Evaluate DeepONet Hybrid Predictions
    u_base_final = get_hybrid_u_base(beta_hyb)
    pred_hyb = []
    with torch.no_grad():
        for i in range(ne):
            net = solver_hyb.models[i]['net']
            out = net(TX_eval_t, u=u_base_final[i]).cpu().numpy().reshape(101, 101)
            pred_hyb.append(out)
    pred_hyb = np.stack(pred_hyb, axis=0)

    abs_err_hyb = float(np.sqrt(np.mean((pred_hyb - v_ref) ** 2)))
    rel_err_hyb = abs_err_hyb / ref_norm
    linf_hyb = float(np.max(np.abs(pred_hyb - v_ref)))

    print(f"DeepONet Hybrid Rel L2 Error: {rel_err_hyb * 100:.2f}% | Abs L2: {abs_err_hyb:.4e} | Linf: {linf_hyb:.4e}")

    # ---------------------------------------------------------------------------------
    # COMPREHENSIVE BENCHMARK TABLE
    # ---------------------------------------------------------------------------------
    speedup = time_qgp / max(time_hyb, 1e-6)
    acc_factor = rel_err_qgp / max(rel_err_hyb, 1e-6)

    print("\n" + "=" * 90)
    print("               HEAD-TO-HEAD BENCHMARK & COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Performance Metric':<32} | {'Original QGPINNs':<24} | {'DeepONet Hybrid':<24}")
    print("-" * 90)
    print(f"{'Initialization':<32} | {'Random (Xavier)':<24} | {'Saved Adaptive Weights':<24}")
    print(f"{'Space-Time Relative L2 Error':<32} | {rel_err_qgp * 100:6.2f}% ({rel_err_qgp:.4e})   | {rel_err_hyb * 100:6.2f}% ({rel_err_hyb:.4e})")
    print(f"{'Space-Time Absolute L2 Error':<32} | {abs_err_qgp:.4e}{'':<14} | {abs_err_hyb:.4e}{'':<14}")
    print(f"{'Space-Time Linf Error':<32} | {linf_qgp:.4e}{'':<14} | {linf_hyb:.4e}{'':<14}")
    print(f"{'Total Optimization Time':<32} | {time_qgp:6.2f} seconds{'':<10} | {time_hyb:6.2f} seconds{'':<10}")
    print(f"{'Accuracy Improvement Factor':<32} | {'Baseline (1.0x)':<24} | {acc_factor:5.1f}x More Accurate")
    print(f"{'Computation Speedup Factor':<32} | {'Baseline (1.0x)':<24} | {speedup:5.1f}x Faster Convergence")
    print("=" * 90)

    # ---------------------------------------------------------------------------------
    # VISUALIZATION: 4-PANEL PUBLICATION COMPARISON PLOT
    # ---------------------------------------------------------------------------------
    save_path = 'results/qgpinns_vs_deeponet_hybrid_comparison.png'
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # Panel A: Spatial profiles across graph at t = 0.50
    ax_a = fig.add_subplot(gs[0, 0])
    t_idx = 50
    x_local = np.linspace(0, 1, 101)
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#ff7f0e']

    for e in range(ne):
        x_e = x_local + e
        ref_e = v_ref[e, :, t_idx]
        qgp_e = pred_qgp[e, :, t_idx]
        hyb_e = pred_hyb[e, :, t_idx]

        ax_a.plot(x_e, ref_e, color='black', linestyle='-', linewidth=2.0,
                  label='FVM Ground Truth' if e == 0 else '')
        ax_a.plot(x_e, qgp_e, color='#d62728', linestyle='--', linewidth=1.8, alpha=0.85,
                  label='Original QGPINNs' if e == 0 else '')
        ax_a.plot(x_e, hyb_e, color='#1f77b4', linestyle=':', linewidth=2.4, alpha=0.9,
                  label='DeepONet Hybrid' if e == 0 else '')

    for b in range(1, ne):
        ax_a.axvline(b, color='#cccccc', linestyle=':', linewidth=0.8)

    ax_a.set_title("A. Solution Profiles along Metric Graph (t = 0.50)", fontsize=13, fontweight='bold', pad=8)
    ax_a.set_xlabel("Unrolled Graph Coordinate (Edges 0 to 6)", fontsize=11)
    ax_a.set_ylabel("Concentration $\\rho(t, x)$", fontsize=11)
    ax_a.set_ylim(-0.05, 1.1)
    ax_a.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9.5)
    ax_a.grid(True, linestyle=':', alpha=0.5)

    # Panel B: Pointwise Absolute Errors |u - u_ref| at t = 0.50
    ax_b = fig.add_subplot(gs[0, 1])
    for e in range(ne):
        x_e = x_local + e
        err_qgp_e = np.abs(pred_qgp[e, :, t_idx] - v_ref[e, :, t_idx])
        err_hyb_e = np.abs(pred_hyb[e, :, t_idx] - v_ref[e, :, t_idx])

        ax_b.plot(x_e, err_qgp_e, color='#d62728', linestyle='-', linewidth=1.8,
                  label='Original QGPINNs Error' if e == 0 else '')
        ax_b.plot(x_e, err_hyb_e, color='#1f77b4', linestyle='-', linewidth=2.0,
                  label='DeepONet Hybrid Error' if e == 0 else '')

    for b in range(1, ne):
        ax_b.axvline(b, color='#cccccc', linestyle=':', linewidth=0.8)

    ax_b.set_title("B. Pointwise Absolute Error $|\hat{\\rho} - \\rho_{FVM}|$ (t = 0.50)", fontsize=13, fontweight='bold', pad=8)
    ax_b.set_xlabel("Unrolled Graph Coordinate (Edges 0 to 6)", fontsize=11)
    ax_b.set_ylabel("Absolute Error", fontsize=11)
    ax_b.set_yscale('log')
    ax_b.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9.5)
    ax_b.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Loss vs Iterations / Convergence Trajectory
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.semilogy(range(1, len(loss_history_qgp) + 1), loss_history_qgp, color='#d62728', linewidth=1.8, alpha=0.85, label='Original QGPINNs (From Scratch)')
    ax_c.semilogy(range(1, len(loss_history_hyb) + 1), loss_history_hyb, color='#1f77b4', linewidth=2.0, alpha=0.9, label='DeepONet Hybrid (Warm-Start)')
    ax_c.set_title("C. Optimization Loss Convergence Trajectory", fontsize=13, fontweight='bold', pad=8)
    ax_c.set_xlabel("Optimization Iterations (Adam)", fontsize=11)
    ax_c.set_ylabel("Total Weighted Residual Loss", fontsize=11)
    ax_c.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9.5)
    ax_c.grid(True, linestyle=':', alpha=0.5)

    # Panel D: Bar Chart: Accuracy & Time Comparison
    ax_d = fig.add_subplot(gs[1, 1])
    categories = ['Relative L2 Error (%)', 'Training Time (sec / 10)']
    qgp_vals = [rel_err_qgp * 100, time_qgp / 10.0]
    hyb_vals = [rel_err_hyb * 100, time_hyb / 10.0]

    x_bars = np.arange(len(categories))
    width = 0.32

    rects1 = ax_d.bar(x_bars - width/2, qgp_vals, width, label='Original QGPINNs', color='#d62728', alpha=0.85, edgecolor='black')
    rects2 = ax_d.bar(x_bars + width/2, hyb_vals, width, label='DeepONet Hybrid', color='#1f77b4', alpha=0.85, edgecolor='black')

    ax_d.set_title(f"D. Quantitative Error & Runtime Comparison (Speedup: {speedup:.1f}x)", fontsize=13, fontweight='bold', pad=8)
    ax_d.set_xticks(x_bars)
    ax_d.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax_d.set_ylabel("Value", fontsize=11)
    ax_d.legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9.5)
    ax_d.grid(True, axis='y', linestyle=':', alpha=0.5)

    # Add labels on bars
    for r in rects1:
        h = r.get_height()
        ax_d.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for r in rects2:
        h = r.get_height()
        ax_d.annotate(f'{h:.2f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle("Head-to-Head Benchmark: Original QGPINNs vs Pretrained DeepONet Hybrid\nChain Graph G1 Space-Time Drift-Diffusion", fontsize=15, fontweight='bold', y=0.99)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to: {save_path}")


if __name__ == '__main__':
    main()

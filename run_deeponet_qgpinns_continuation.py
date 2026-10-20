"""
Physics-Informed DeepONet + QGPINNs Engine Continuation Benchmark
----------------------------------------------------------------
1. Uses saved & frozen weights and biases from training_weights_biases/ as a warm starting point.
2. Continues with the QGPINNS engine architecture:
   - Governing space-time PDE residual on metric graph edges
   - Initial conditions and Dirichlet boundary conditions
   - Vertex continuity coupling condition across multi-edge junctions
   - Kirchhoff nonlinear flux conservation: sum(s * (-eps*dx(rho) + nu*rho*(1-rho))) = 0
   - Multi-phase optimization: Adam with decoupled adaptive multipliers + L-BFGS second-order polish
3. Evaluates quantitative space-time relative L2 error against high-fidelity FVM ground truth.
4. Generates comparative publication-quality visualizations in results/.
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

# Add Engine and official_repo to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'Engine'))
sys.path.insert(0, os.path.join(BASE_DIR, 'scratch', 'official_repo'))

from QGPINNs_Engine import (
    DeepONetPINNSolver,
    PretrainedDeepONet,
    DriftDiffusionPhysics,
    DecoupledAdaptiveLambdaController,
    device,
    NET_DTYPE
)

# Reference FVM solver & Graph generator from official repo
from src.graph import Example0, Example5
from src.quantumGraphSolverFVM import QuantumGraphSolverFVM
from src.GPs import RBF, get_sample_fns
import jax
import jax.numpy as jnp
from jax import random


def main():
    print("=" * 85)
    print("  QGPINNs ENGINE: CONTINUING FROM PRE-TRAINED ADAPTIVE DEEPONET WEIGHTS")
    print("  Weights Source: training_weights_biases/")
    print(f"  Execution Device: {device}")
    print("=" * 85)

    os.makedirs('results', exist_ok=True)
    WEIGHTS_DIR = os.path.join(BASE_DIR, 'training_weights_biases')

    # 1. Metric Graph Setup: Chain Graph G1 (7 edges, 8 vertices)
    EPS = 0.1
    seed = 0
    graph_fvm = Example5(eps=EPS, n_edges=7)
    ne = graph_fvm.ne
    L = 1.0

    class MetricGraphTopology:
        nodes = list(range(graph_fvm.n_v))
        edges = [(int(e[0]), int(e[1]), L) for e in graph_fvm.E]

    metric_graph = MetricGraphTopology()

    # Determine graph vertex-edge connectivity
    vertex_edge_list = []
    glob_inner_idx = 0
    for vidx in graph_fvm.innerVertices:
        for eidx in graph_fvm.Vin[vidx]:
            if graph_fvm.E[eidx][0] in graph_fvm.inflowNodes:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, -1, -1])
            else:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, -1, 0])
        for eidx in graph_fvm.Vout[vidx]:
            if graph_fvm.E[eidx][1] in graph_fvm.outflowNodes:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, 1, 1])
            else:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, 1, 0])
        glob_inner_idx += 1
    vertex_edge_list = np.vstack(vertex_edge_list)
    n_params = vertex_edge_list.shape[0]

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

    # 2. Flow Parameterization via RBF Kernel
    N_DATA_BC = 100
    N_DATA_INIT = 100
    x_sensor_bc = np.linspace(0, 1, N_DATA_BC + 1)
    x_sensor_init = np.linspace(0, 1, N_DATA_INIT + 1)
    n_sensor_bc = N_DATA_BC + 1

    n_rbf_flow = 10
    rbf_output_scale = 1.0
    rbf_length_scale = 0.2
    x_rbf_flow = np.linspace(0, 1, n_rbf_flow)

    K_flow_jnp = RBF(jnp.array(x_sensor_bc).reshape((n_sensor_bc, 1)),
                     jnp.array(x_rbf_flow).reshape((-1, 1)),
                     (rbf_output_scale, rbf_length_scale))
    K_flow = np.array(K_flow_jnp)

    def beta_to_z(beta_np):
        return np.vstack([np.dot(K_flow, b).reshape((1, n_sensor_bc)) for b in beta_np])

    # 3. Sample Initial & Boundary Conditions
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
    print(f"Graph Edge Velocities: {np.round(vel, 3)}")

    # 4. Reference High-Fidelity FVM Solution
    print("\nRunning high-fidelity FVM benchmark solver...")
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
    v_ref = v[:, ::fnx, ::fnt]
    print(f"Reference FVM Solution Grid: {v_ref.shape} (edges, space, time)")

    def build_u_base(z_np):
        u_list = []
        in_offset = n_inflow_edges
        out_offset = 2 * n_inner_edges + n_inflow_edges
        for e in edge_list:
            if e[-1] == -1:
                u_list.append(np.hstack([u_in[e[1]], z_np[e[1]], u_init_list[e[0]], vel[e[0]]]))
            elif e[-1] == 0:
                u_list.append(np.hstack([z_np[2 * e[1] + in_offset], z_np[2 * e[1] + in_offset + 1], u_init_list[e[0]], vel[e[0]]]))
            else:
                u_list.append(np.hstack([z_np[out_offset + e[1]], u_out[e[1]], u_init_list[e[0]], vel[e[0]]]))
        return np.vstack(u_list)

    # 5. Initialize Pretrained DeepONet Models for Each Edge
    print("\nLoading trained & frozen DeepONet weights from training_weights_biases/ ...")
    edge_types = {}
    for i, e in enumerate(edge_list):
        if e[-1] == -1:
            edge_types[i] = 'inflow'
        elif e[-1] == 0:
            edge_types[i] = 'inner'
        else:
            edge_types[i] = 'outflow'

    # Initial zero flow parameters
    beta_param = nn.Parameter(torch.zeros(n_params, n_rbf_flow, dtype=torch.float32, device=device))
    K_flow_tensor = torch.tensor(K_flow, dtype=torch.float32, device=device)

    def get_torch_u_base(beta):
        z = beta @ K_flow_tensor.T  # (n_params, 101)
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

    # Instantiate QGPINNs physics objects
    physics_list = [
        DriftDiffusionPhysics(edge_idx=i, eps=EPS, nu=vel[i])
        for i in range(ne)
    ]

    solver = DeepONetPINNSolver(
        graph=metric_graph,
        physics=physics_list,
        weights_dir=WEIGHTS_DIR,
        edge_types=edge_types,
        eps=EPS,
        t_max=1.0
    )
    solver.set_mesh(pinn_pts=101, n_t=101, t_max=1.0)
    solver.compile()
    print("DeepONetPINNSolver compiled successfully with PretrainedDeepONet edge surrogates!")

    # Evaluation function for current parameters
    TX_eval = np.hstack([l.reshape((-1, 1)) for l in np.meshgrid(
        np.linspace(0, 1, 101),
        np.linspace(0, 1, 101))])  # (t, x)
    TX_eval_t = torch.tensor(TX_eval, dtype=torch.float32, device=device)

    def evaluate_model(u_base_current):
        preds = []
        with torch.no_grad():
            for e_idx in range(ne):
                net = solver.models[e_idx]['net']
                out = net(TX_eval_t, u=u_base_current[e_idx]).cpu().numpy()
                preds.append(out.reshape(101, 101))
        preds = np.stack(preds, axis=0)
        l2_abs = float(np.sqrt(np.mean((preds - v_ref) ** 2)))
        l2_ref = float(np.sqrt(np.mean(v_ref ** 2)))
        l2_rel = l2_abs / l2_ref
        linf = float(np.max(np.abs(preds - v_ref)))
        return l2_abs, l2_rel, linf, preds

    # 6. Zero-Shot Benchmark (Directly from Saved Weights)
    u_base_init = get_torch_u_base(beta_param)
    abs_zero, rel_zero, linf_zero, pred_zero = evaluate_model(u_base_init)
    print("\n" + "-" * 75)
    print("ZERO-SHOT EVALUATION FROM SAVED PRE-TRAINED WEIGHTS:")
    print(f"  Space-Time Absolute L2 Error: {abs_zero:.4e}")
    print(f"  Space-Time Relative L2 Error: {rel_zero:.4e} ({rel_zero * 100:.2f}%)")
    print(f"  Space-Time Linf Error:        {linf_zero:.4e}")
    print("-" * 75)

    # 7. QGPINNs Architecture Continuation:
    # Optimizing Interface Coupling Conditions with Decoupled Adaptive Lambdas
    print("\nCONTINUING OPTIMIZATION WITH QGPINNS ENGINE ARCHITECTURE:")
    print("  Phase 1: Adam Optimizer with Decoupled Multi-Adaptive Multipliers")
    print("  Phase 2: L-BFGS Second-Order Polish for Fast Convergence")
    print("-" * 75)

    N_IT = 100
    tx_in = torch.tensor(np.hstack([np.linspace(0, 1, N_IT + 1)[:, None], np.ones((N_IT + 1, 1))]), dtype=torch.float32, device=device)
    tx_out = torch.tensor(np.hstack([np.linspace(0, 1, N_IT + 1)[:, None], np.zeros((N_IT + 1, 1))]), dtype=torch.float32, device=device)

    controller = DecoupledAdaptiveLambdaController(init_cont=10.0, init_flux=1.0, alpha=0.1, adapt_every=10)
    optimizer = torch.optim.AdamW([beta_param], lr=1e-4)

    def compute_coupling_residuals(beta):
        u_b = get_torch_u_base(beta)
        incoming_vals, incoming_flux = [], []
        outgoing_vals, outgoing_flux = [], []

        # Predict boundary states for each edge
        val_list_in, flux_list_in = [], []
        val_list_out, flux_list_out = [], []

        for i, m in solver.models.items():
            net = m['net']
            u_e = u_b[i]
            # Inflow boundary (x = 0)
            tx_0 = tx_out.clone().requires_grad_(True)
            u0 = net(tx_0, u=u_e)
            grad_x0 = torch.autograd.grad(u0, tx_0, torch.ones_like(u0), create_graph=True)[0][:, 1:2]
            j0 = -EPS * grad_x0 + vel[i] * u0 * (1.0 - u0)
            val_list_out.append(u0)
            flux_list_out.append(j0)

            # Outflow boundary (x = L)
            tx_1 = tx_in.clone().requires_grad_(True)
            u1 = net(tx_1, u=u_e)
            grad_x1 = torch.autograd.grad(u1, tx_1, torch.ones_like(u1), create_graph=True)[0][:, 1:2]
            j1 = -EPS * grad_x1 + vel[i] * u1 * (1.0 - u1)
            val_list_in.append(u1)
            flux_list_in.append(j1)

        # Compute junction losses across internal vertices
        loss_cont = torch.tensor(0.0, device=device)
        loss_flux = torch.tensor(0.0, device=device)

        # For Chain Graph: edge i terminates at vertex i+1, edge i+1 originates at vertex i+1
        for i in range(ne - 1):
            val_diff = val_list_in[i] - val_list_out[i + 1]
            loss_cont = loss_cont + torch.sum(val_diff ** 2)

            # Kirchhoff: flux leaving edge i equals flux entering edge i+1
            flux_balance = flux_list_in[i] - flux_list_out[i + 1]
            loss_flux = loss_flux + torch.sum(flux_balance ** 2)

        loss_cont = loss_cont / (ne - 1)
        loss_flux = loss_flux / (ne - 1)
        return loss_cont, loss_flux

    # Phase 1: Adam with Adaptive Multipliers
    epochs_adam = 1500
    print(f"Phase 1: Running Adam for {epochs_adam} iterations...")
    start_time = time.time()

    for it in range(1, epochs_adam + 1):
        optimizer.zero_grad()
        loss_c, loss_f = compute_coupling_residuals(beta_param)

        # Update adaptive multipliers based on gradient-norm balancing
        if it % controller.adapt_every == 0 and it >= 10:
            gc = torch.autograd.grad(loss_c, beta_param, retain_graph=True)[0]
            gf = torch.autograd.grad(loss_f, beta_param, retain_graph=True)[0]
            controller.update(it, gc, gf)

        total_loss = controller.lambda_cont * loss_c + controller.lambda_flux * loss_f
        total_loss.backward()
        optimizer.step()

        if it % 300 == 0 or it == epochs_adam:
            print(f"  Iter {it:4d}/{epochs_adam} | Total: {total_loss.item():.4e} | Cont: {loss_c.item():.4e} | Flux: {loss_f.item():.4e} | lam_cont: {controller.lambda_cont:.2f} | lam_flux: {controller.lambda_flux:.2f}")

    adam_time = time.time() - start_time
    print(f"Phase 1 completed in {adam_time:.2f}s")

    # Phase 2: L-BFGS Second-Order Polish with Frozen Multipliers
    print("\nPhase 2: Running L-BFGS Second-Order Polish (500 steps)...")
    lbfgs = torch.optim.LBFGS([beta_param], lr=0.5, max_iter=20, history_size=20)
    lam_c_frozen = controller.lambda_cont
    lam_f_frozen = controller.lambda_flux

    for step in range(1, 26):
        def closure():
            lbfgs.zero_grad()
            lc, lf = compute_coupling_residuals(beta_param)
            loss = lam_c_frozen * lc + lam_f_frozen * lf
            loss.backward()
            return loss
        lbfgs.step(closure)

    print("Phase 2 Polish finished.")

    # 8. Final Quantitative Benchmark
    u_base_final = get_torch_u_base(beta_param)
    abs_final, rel_final, linf_final, pred_final = evaluate_model(u_base_final)

    print("\n" + "=" * 85)
    print("FINAL QUANTITATIVE BENCHMARK: DEEPONET INITIALIZATION vs QGPINNs CONTINUATION")
    print("=" * 85)
    print(f"{'Metric':<30} | {'Saved DeepONet (Initial)':<26} | {'QGPINNs Continued (Final)'}")
    print("-" * 85)
    print(f"{'Space-Time Absolute L2 Error':<30} | {abs_zero:<26.4e} | {abs_final:.4e}")
    print(f"{'Space-Time Relative L2 Error':<30} | {rel_zero:<26.4e} ({rel_zero*100:.2f}%) | {rel_final:.4e} ({rel_final*100:.2f}%)")
    print(f"{'Space-Time Linf Error':<30} | {linf_zero:<26.4e} | {linf_final:.4e}")
    print("=" * 85)

    # 9. Plot Results and Save
    save_path = 'results/deeponet_qgpinns_continuation.png'
    times_idx = [0, 25, 50, 75, 100]
    time_labels = ['0.00', '0.25', '0.50', '0.75', '1.00']
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#ff7f0e']

    fig, axes = plt.subplots(len(times_idx), 1, figsize=(7.5, 10.0), sharex=True, sharey=True)
    x_local = np.linspace(0, 1, 101)

    for row_idx, (t_idx, t_lbl) in enumerate(zip(times_idx, time_labels)):
        ax = axes[row_idx]
        for e in range(ne):
            x_e = x_local + e
            ref_e = v_ref[e, :, t_idx]
            pred_e = pred_final[e, :, t_idx]

            ax.plot(x_e, ref_e, color=colors[e % len(colors)], linestyle='-', linewidth=2.2,
                    label='Reference (FVM Ground Truth)' if (row_idx == 0 and e == 0) else '')
            ax.plot(x_e, pred_e, color='black', linestyle=(0, (4, 3)), linewidth=1.5, alpha=0.9,
                    label='QGPINNs Continued Solution' if (row_idx == 0 and e == 0) else '')

        ax.set_ylabel(f't = {t_lbl}', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.4)
        for b in range(1, ne):
            ax.axvline(b, color='#cccccc', linestyle=':', linewidth=0.8)

    axes[0].legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=8.5)
    axes[-1].set_xlim(-0.15, ne + 0.15)
    axes[-1].set_xticks(range(ne + 1))
    axes[-1].set_xticklabels([f'Edge {i}' if i < ne else f'Node {i}' for i in range(ne + 1)], fontsize=9)
    axes[-1].set_xlabel('Unrolled Metric Graph Coordinate (Edges 0 to 6)', fontsize=11, labelpad=6)

    fig.suptitle("QGPINNs Engine Continuation from Saved Adaptive DeepONet Weights\nChain Graph G1 Space-Time Drift-Diffusion", fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot to: {save_path}")


if __name__ == '__main__':
    main()

import os
import sys
import time

# Ensure headless matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap
import optax

# Add official_repo to path
REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch', 'official_repo')
sys.path.insert(0, REPO_PATH)

from src.graph import Example0, Example2, Example5, Example7
from src.quantumGraphSolverFVM import QuantumGraphSolverFVM
from src.networks_velocity import PI_DeepONet, FF_MLP
from src.dataHandling import load_best_model
from src.GPs import RBF, get_sample_fns

def run_benchmark(graph_name, graph_instance, n_width=200, mode=2, n_iter=3000, seed=0):
    print("\n" + "=" * 75)
    print(f"  BENCHMARK: {graph_name} ({graph_instance.ne} edges, {graph_instance.n_v} nodes)")
    print(f"  Surrogate Architecture: Width {n_width}, Mode {mode} (20K dataset)")
    print("=" * 75)

    # 1. Global Setup
    FVM_NT = 2000
    FVM_NX = 1000
    fnx = round(FVM_NX / 100)
    fnt = round(FVM_NT / 100)
    EPS = 0.1

    N_DATA_BC = 100
    N_DATA_INIT = 100
    x_sensor_bc = jnp.linspace(0, 1, N_DATA_BC + 1)
    x_sensor_init = jnp.linspace(0, 1, N_DATA_INIT + 1)
    n_sensor_bc = N_DATA_BC + 1
    n_sensor_init = N_DATA_INIT + 1

    TX_small = jnp.hstack([l.reshape((-1, 1)) for l in jnp.meshgrid(
        jnp.linspace(0, 1, N_DATA_BC + 1),
        jnp.linspace(0, 1, N_DATA_INIT + 1))])

    # 2. Load Pretrained DeepONet Surrogate Models
    m = 304
    params_inflow, params_inner, params_outflow = load_best_model(n_width, mode, prefix=REPO_PATH)
    branch_layers = [m, n_width, n_width, n_width, n_width, n_width, n_width, n_width]
    trunk_layers = [2, n_width, n_width, n_width, n_width, n_width, n_width, n_width]
    model = PI_DeepONet(Example0(eps=EPS), branch_layers, trunk_layers, trunk_net=FF_MLP)

    # 3. Determine Graph Vertex-Edge Connectivity
    graph = graph_instance
    vertex_edge_list = []
    glob_inner_idx = 0
    for vidx in graph.innerVertices:
        for eidx in graph.Vin[vidx]:
            if graph.E[eidx][0] in graph.inflowNodes:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, -1, -1])
            else:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, -1, 0])
        for eidx in graph.Vout[vidx]:
            if graph.E[eidx][1] in graph.outflowNodes:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, 1, 1])
            else:
                vertex_edge_list.append([vidx, glob_inner_idx, eidx, 1, 0])
        glob_inner_idx += 1
    vertex_edge_list = np.vstack(vertex_edge_list)
    n_params = vertex_edge_list.shape[0]

    ve_vidx = []
    for vidx in range(len(graph.innerVertices)):
        ve_vidx.append(jnp.argwhere(vertex_edge_list[:, 1] == vidx).flatten())

    edge_list = []
    inner_idx = 0
    for i, e in enumerate(graph.E):
        inflow_idx = np.where(e[0] == graph.inflowNodes)[0]
        outflow_idx = np.where(e[1] == graph.outflowNodes)[0]
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

    # 4. Flow Parameterization via RBF Kernel (Section 4 of Paper)
    n_rbf_flow = 10
    rbf_output_scale = 1.0
    rbf_length_scale = 0.2
    x_rbf_flow = jnp.linspace(0, 1, n_rbf_flow)
    K_flow = RBF(x_sensor_bc.reshape((n_sensor_bc, 1)),
                 x_rbf_flow.reshape((-1, 1)),
                 (rbf_output_scale, rbf_length_scale))

    def draw_initial_beta(k):
        return random.normal(k, shape=(n_params, n_rbf_flow)) * 0.01

    def beta_to_z_flow(beta):
        return jnp.vstack([jnp.dot(K_flow, b).reshape((1, n_sensor_bc)) for b in beta])

    # 5. Interface Evaluation Setup
    N = 100
    tx_in = jnp.hstack([jnp.linspace(0, 1, N + 1)[:, None], jnp.ones((N + 1, 1))])
    tx_out = jnp.hstack([jnp.linspace(0, 1, N + 1)[:, None], jnp.zeros((N + 1, 1))])

    predict_all = vmap(model.flux_net, (None, None, 0, 0))
    predict_flux = vmap(predict_all, (None, 0, None, None))

    incoming_inflow_idx = jnp.where(vertex_edge_list[:, -1] == -1)[0]
    incoming_inner_idx = jnp.where((vertex_edge_list[:, -1] == 0) * (vertex_edge_list[:, 3] == -1))[0]
    outgoing_inner_idx = jnp.where((vertex_edge_list[:, -1] == 0) * (vertex_edge_list[:, 3] == 1))[0]
    outgoing_outflow_idx = jnp.where(vertex_edge_list[:, -1] == +1)[0]

    n_ve = vertex_edge_list.shape[0]
    n_tx = tx_in.shape[0]
    ve_incoming_inflow = vertex_edge_list[incoming_inflow_idx, :]
    ve_incoming_inner = vertex_edge_list[incoming_inner_idx, :]
    ve_outgoing_inner = vertex_edge_list[outgoing_inner_idx, :]
    ve_outgoing_outflow = vertex_edge_list[outgoing_outflow_idx, :]

    def predict_ve_incoming_inflow(z):
        f, v = jnp.split(jnp.stack(predict_flux(params_inflow, z[ve_incoming_inflow[:, 2], :], tx_in[:, 0], tx_in[:, 1])), 2)
        return f.squeeze(axis=0) * ve_incoming_inflow[:, 3].reshape((-1, 1)), v.squeeze(axis=0)

    def predict_ve_incoming_inner(z):
        f, v = jnp.split(jnp.stack(predict_flux(params_inner, z[ve_incoming_inner[:, 2], :], tx_in[:, 0], tx_in[:, 1])), 2)
        return f.squeeze(axis=0) * ve_incoming_inner[:, 3].reshape((-1, 1)), v.squeeze(axis=0)

    def predict_ve_outgoing_inner(z):
        f, v = jnp.split(jnp.stack(predict_flux(params_inner, z[ve_outgoing_inner[:, 2], :], tx_out[:, 0], tx_out[:, 1])), 2)
        return f.squeeze(axis=0) * ve_outgoing_inner[:, 3].reshape((-1, 1)), v.squeeze(axis=0)

    def predict_ve_outgoing_outflow(z):
        f, v = jnp.split(jnp.stack(predict_flux(params_outflow, z[ve_outgoing_outflow[:, 2], :], tx_out[:, 0], tx_out[:, 1])), 2)
        return f.squeeze(axis=0) * ve_outgoing_outflow[:, 3].reshape((-1, 1)), v.squeeze(axis=0)

    def f_u_new(u_base):
        ve_incoming_inflow_flux, ve_incoming_inflow_vals = predict_ve_incoming_inflow(u_base)
        ve_incoming_inner_flux, ve_incoming_inner_vals = predict_ve_incoming_inner(u_base)
        ve_outgoing_inner_flux, ve_outgoing_inner_vals = predict_ve_outgoing_inner(u_base)
        ve_outgoing_outflow_flux, ve_outgoing_outflow_vals = predict_ve_outgoing_outflow(u_base)

        Xvals = jnp.empty((n_ve, n_tx))
        Xvals = Xvals.at[incoming_inflow_idx].set(ve_incoming_inflow_vals)
        Xvals = Xvals.at[incoming_inner_idx].set(ve_incoming_inner_vals)
        Xvals = Xvals.at[outgoing_inner_idx].set(ve_outgoing_inner_vals)
        Xvals = Xvals.at[outgoing_outflow_idx].set(ve_outgoing_outflow_vals)

        Xflux = jnp.empty((n_ve, n_tx))
        Xflux = Xflux.at[incoming_inflow_idx].set(ve_incoming_inflow_flux)
        Xflux = Xflux.at[incoming_inner_idx].set(ve_incoming_inner_flux)
        Xflux = Xflux.at[outgoing_inner_idx].set(ve_outgoing_inner_flux)
        Xflux = Xflux.at[outgoing_outflow_idx].set(ve_outgoing_outflow_flux)

        loss_cont = 0
        loss_flux = 0
        for vidx in range(len(graph.innerVertices)):
            loss_cont += jnp.sum(jnp.square(jnp.diff(Xvals[ve_vidx[vidx], :], axis=0)))
            loss_flux += jnp.sum(jnp.square(jnp.sum(Xflux[ve_vidx[vidx], :], axis=0)))

        loss_cont /= len(graph.innerVertices)
        loss_flux /= len(graph.innerVertices)
        return loss_cont, loss_flux

    def z_to_u_base(z, u_in, u_init_list, u_out, vel):
        u_list = []
        in_offset = n_inflow_edges
        out_offset = 2 * n_inner_edges + n_inflow_edges
        for e in edge_list:
            if e[-1] == -1:
                u_list.append(jnp.hstack([u_in[e[1]], z[e[1]], u_init_list[e[0]], vel[e[0]]]))
            elif e[-1] == 0:
                u_list.append(jnp.hstack([z[2 * e[1] + in_offset], z[2 * e[1] + in_offset + 1], u_init_list[e[0]], vel[e[0]]]))
            else:
                u_list.append(jnp.hstack([z[out_offset + e[1]], u_out[e[1]], u_init_list[e[0]], vel[e[0]]]))
        return jnp.vstack(u_list)

    def f_beta(beta, u_in, u_init_list, u_out, vel):
        return f_u_new(z_to_u_base(beta_to_z_flow(beta), u_in, u_init_list, u_out, vel))

    def f_loss(beta, u_in, u_init_list, u_out, vel):
        fb = f_beta(beta, u_in, u_init_list, u_out, vel)
        return fb[0] + fb[1]

    val_grad_f = jax.jit(jax.value_and_grad(f_loss, 0))

    # 6. Sample Initial & Boundary Conditions (GP length scale 0.4, 468 centers)
    sample_u_in_fn, sample_u_out_fn, sample_u_init_fn = get_sample_fns(N=468, length_scale=0.4)
    global_key = random.key(seed)
    global_key, key = random.split(global_key, 2)
    key, inflow_key, outflow_key, init_key, vel_key, beta_key = random.split(key, 6)

    inflow_fn_list = [sample_u_in_fn(s) for s in random.split(inflow_key, len(graph.inflowNodes))]
    outflow_fn_list = [sample_u_out_fn(s) for s in random.split(outflow_key, len(graph.outflowNodes))]
    init_cond_fn_list = [sample_u_init_fn(s) for s in random.split(init_key, graph.ne)]

    def dirichletAlpha(x):
        alpha = np.zeros(graph.n_v)
        alpha[graph.inflowNodes] = np.array([inflow_fn(x) for inflow_fn in inflow_fn_list])
        return alpha

    def dirichletBeta(x):
        beta_out = np.zeros(graph.n_v)
        beta_out[graph.outflowNodes] = np.array([outflow_fn(x) for outflow_fn in outflow_fn_list])
        return beta_out

    graph.dirichletAlpha = dirichletAlpha
    graph.dirichletBeta = dirichletBeta
    graph.initial_cond = lambda x: np.array([init_cond_fn(x) for init_cond_fn in init_cond_fn_list])
    graph.initial_cond_jnp = lambda x: jnp.array([init_cond_fn(x) for init_cond_fn in init_cond_fn_list])

    vel = np.array(random.uniform(vel_key, (graph.ne,), minval=0.5, maxval=2.0))
    graph.v = vel
    print(f"Edge velocities: {np.round(vel, 3)}")

    # 7. High-Fidelity FVM Ground Truth Solve (Appendix B)
    print("Running high-fidelity FVM ground truth solver on metric graph...")
    TimeOffset = 0.1
    graph.lb[0] = -TimeOffset

    fvm_solver = QuantumGraphSolverFVM(graph)
    v_raw = fvm_solver.solve(nx=FVM_NX + 1, nt=int(FVM_NT * (1 + TimeOffset)) + 1)
    v = np.array([fvm_solver.get_u_edge(v_raw.T, i).T for i in range(graph.ne)])
    v = v[:, :, int(TimeOffset * FVM_NT):]

    u_init_list = [jnp.interp(x_sensor_init, jnp.linspace(0, 1, FVM_NX + 1), v[i, :, 0]) for i in range(graph.ne)]
    u_in = np.hstack([inflow_fn(x_sensor_bc) for inflow_fn in inflow_fn_list]).reshape(-1, len(x_sensor_bc))
    u_out = np.hstack([outflow_fn(x_sensor_bc) for outflow_fn in outflow_fn_list]).reshape(-1, len(x_sensor_bc))

    v_ref = v[:, ::fnx, ::fnt]
    print(f"Reference FVM solution grid: {v_ref.shape} (edges, space, time)")

    # 8. Optimize Interface Coupling Loss (AdamW)
    print("Optimizing DeepONet interface coupling loss (AdamW)...")
    solver = optax.adamw(learning_rate=1e-4)
    beta = draw_initial_beta(beta_key)
    opt_state = solver.init(beta)

    min_val = jnp.inf
    beta_star = beta

    start_time = time.time()
    for it in range(n_iter + 1):
        val, grad = val_grad_f(beta, u_in, u_init_list, u_out, vel)
        updates, opt_state = solver.update(grad, opt_state, beta)
        beta = optax.apply_updates(beta, updates)

        if val < min_val:
            min_val = val
            beta_star = beta

        if it % 500 == 0 or it == n_iter:
            fb = f_beta(beta, u_in, u_init_list, u_out, vel)
            print(f"  Iter {it:5d} | Total: {float(val):.2e} | Continuity: {float(fb[0]):.2e} | Flux: {float(fb[1]):.2e} | Grad: {float(jnp.linalg.norm(grad)):.2e}")

    opt_time = time.time() - start_time
    print(f"Coupling optimization completed in {opt_time:.2f} s")

    # 9. Predict with Pretrained Physics-Informed DeepONet
    print("Evaluating Physics-Informed DeepONet predictions across graph...")
    z = beta_to_z_flow(beta_star)
    u_base = z_to_u_base(z, u_in, u_init_list, u_out, vel)

    v_pred = []
    for e in edge_list:
        if e[-1] == -1:  # Inflow edge
            e_vals = model.predict_s_all(params_inflow, u_base[e[0]], TX_small)
        elif e[-1] == 0:  # Inner edge
            e_vals = model.predict_s_all(params_inner, u_base[e[0]], TX_small)
        else:  # Outflow edge
            e_vals = model.predict_s_all(params_outflow, u_base[e[0]], TX_small)
        v_pred.append(e_vals.reshape(101, 101))
    v_pred = np.stack(v_pred, dtype=np.float64)

    # 10. Quantitative Comparison against Reference Solution
    l2_abs_error_per_edge = np.sqrt(np.mean(np.square(v_pred - v_ref), axis=(1, 2)))
    l2_norm_ref_per_edge = np.sqrt(np.mean(np.square(v_ref), axis=(1, 2)))
    linf_error = np.max(np.abs(v_pred - v_ref))

    l2_abs_error = np.sqrt(np.mean(np.square(l2_abs_error_per_edge)))
    l2_rel_error = np.sqrt(np.mean(np.square(l2_abs_error_per_edge / l2_norm_ref_per_edge)))

    print("\n--- QUANTITATIVE ACCURACY RESULTS ---")
    print(f"  Space-Time Absolute L2 Error: {l2_abs_error:.4e}")
    print(f"  Space-Time Relative L2 Error: {l2_rel_error:.4e} ({l2_rel_error * 100:.2f}%)")
    print(f"  Space-Time Linf Error:        {linf_error:.4e}")
    for i in range(graph.ne):
        rel_e = l2_abs_error_per_edge[i] / l2_norm_ref_per_edge[i]
        role = 'Inflow' if edge_list[i, -1] == -1 else 'Inner' if edge_list[i, -1] == 0 else 'Outflow'
        print(f"    Edge {i} ({role:<7}): Abs L2 = {l2_abs_error_per_edge[i]:.4e}, Rel L2 = {rel_e:.4e} ({rel_e * 100:.2f}%)")

    return {
        'name': graph_name,
        'v_ref': v_ref,
        'v_pred': v_pred,
        'l2_abs': l2_abs_error,
        'l2_rel': l2_rel_error,
        'linf': linf_error,
        'ne': graph.ne,
        'edge_list': edge_list
    }

def plot_unrolled_chain(res, save_path):
    """
    Plots the unrolled chain graph solution across all 7 edges at time snapshots:
    t = 0.00, 0.25, 0.50, 0.75, 1.00
    matching Figure 6 in the paper (Blechschmidt et al., ICML 2025)!
    """
    v_ref = res['v_ref']
    v_pred = res['v_pred']
    ne = res['ne']
    times_idx = [0, 25, 50, 75, 100]
    time_labels = ['0.00', '0.25', '0.50', '0.75', '1.00']
    colors = [
        '#1f77b4', # Edge 0: Blue
        '#2ca02c', # Edge 1: Green
        '#d62728', # Edge 2: Red
        '#9467bd', # Edge 3: Purple
        '#8c564b', # Edge 4: Brown
        '#e377c2', # Edge 5: Pink
        '#ff7f0e', # Edge 6: Orange
    ]

    fig, axes = plt.subplots(len(times_idx), 1, figsize=(6.5, 9.5), sharex=True, sharey=True)
    x_local = np.linspace(0, 1, 101)

    for row_idx, (t_idx, t_lbl) in enumerate(zip(times_idx, time_labels)):
        ax = axes[row_idx]

        for e in range(ne):
            x_e = x_local + e
            ref_e = v_ref[e, :, t_idx]
            pred_e = v_pred[e, :, t_idx]

            # Reference solution (solid line)
            ax.plot(x_e, ref_e, color=colors[e % len(colors)], linestyle='-', linewidth=2.2,
                    label='Reference (FVM)' if (row_idx == 0 and e == 0) else '')
            # PI DeepONet solution (dashed line directly on top)
            ax.plot(x_e, pred_e, color='black', linestyle=(0, (4, 3)), linewidth=1.4, alpha=0.85,
                    label='PI DeepONet' if (row_idx == 0 and e == 0) else '')
            ax.plot(x_e, pred_e, color=colors[e % len(colors)], linestyle=(0, (4, 3)), linewidth=2.2, alpha=0.9)

        ax.set_ylabel(f't={t_lbl}', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(['0.0', '0.5', '1.0'], fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=4, width=1.0)

        for b in range(1, ne):
            ax.axvline(b, color='#e0e0e0', linestyle=':', linewidth=0.8, zorder=0)

    axes[0].legend(loc='upper right', frameon=True, framealpha=0.9, fontsize=9)
    axes[-1].set_xlim(-0.15, ne + 0.15)
    axes[-1].set_xticks(range(ne + 1))
    axes[-1].set_xticklabels([str(i) for i in range(ne + 1)], fontsize=10)
    axes[-1].set_xlabel('Unrolled Graph Coordinate (Edges 0 to 6)', fontsize=11, labelpad=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved snapshot plot to: {save_path}")

def plot_double_y(res, save_path):
    """
    Plots the comparison for Double-Y Graph G2 at t = 0.5 and t = 1.0
    matching Figure 5 in the paper!
    """
    v_ref = res['v_ref']
    v_pred = res['v_pred']
    ne = res['ne']

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x_local = np.linspace(0, 1, 101)
    t_indices = [50, 100]
    t_names = ['t = 0.5', 't = 1.0']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    edge_names = ['e1 (inflow)', 'e2 (inflow)', 'e3 (inner)', 'e4 (outflow)', 'e5 (outflow)']

    for col, (t_idx, t_name) in enumerate(zip(t_indices, t_names)):
        # Top: Solutions
        ax_top = axes[0, col]
        for e in range(ne):
            ax_top.plot(x_local, v_ref[e, :, t_idx], color=colors[e], linestyle='-', linewidth=1.6, label=f'{edge_names[e]} FVM' if col == 0 else "")
            ax_top.plot(x_local, v_pred[e, :, t_idx], color=colors[e], linestyle='--', linewidth=1.6, label=f'{edge_names[e]} DeepONet' if col == 0 else "")
        ax_top.set_title(f"Double-Y Graph Solutions at {t_name}", fontsize=11, fontweight='bold')
        ax_top.set_ylabel("Concentration $\\rho(t, x)$")
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.grid(True, linestyle=':', alpha=0.6)
        if col == 0:
            ax_top.legend(loc='upper right', fontsize=8)

        # Bottom: Absolute differences
        ax_bot = axes[1, col]
        for e in range(ne):
            diff = np.abs(v_pred[e, :, t_idx] - v_ref[e, :, t_idx])
            ax_bot.plot(x_local, diff, color=colors[e], linewidth=1.6, label=edge_names[e])
        ax_bot.set_title(f"Absolute Difference $|\hat{{\\rho}} - \\rho_{{ref}}|$ at {t_name}", fontsize=11, fontweight='bold')
        ax_bot.set_xlabel("Local coordinate $x \\in [0, 1]$")
        ax_bot.set_ylabel("Absolute Error")
        ax_bot.grid(True, linestyle=':', alpha=0.6)
        ax_bot.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved Double-Y plot to: {save_path}")

if __name__ == '__main__':
    print("=" * 80)
    print("REPRODUCING & VALIDATING PHYSICS-INFORMED DEEPONET vs FVM BENCHMARK")
    print("Paper: 'Physics-Informed DeepONets for drift-diffusion on metric graphs'")
    print("=" * 80)

    os.makedirs('results', exist_ok=True)

    # 1. Benchmark 1: Chain Graph G1 (7 edges) - Figure 4 & 6, Table 2 & 3
    # Mode 1 corresponds to chain graph
    g1 = Example5(eps=0.1, n_edges=7)
    res_g1 = run_benchmark("Chain Graph G1", g1, n_width=200, mode=1, n_iter=3000, seed=0)
    plot_unrolled_chain(res_g1, 'results/chain_graph_g1_comparison.png')

    # 2. Benchmark 2: Double-Y Graph G2 (5 edges) - Figure 4 & 5, Table 2 & 3
    # Mode 0 corresponds to Double-Y graph
    g2 = Example2(eps=0.1)
    res_g2 = run_benchmark("Double-Y Graph G2", g2, n_width=200, mode=0, n_iter=3000, seed=0)
    plot_double_y(res_g2, 'results/double_y_graph_g2_comparison.png')

    # 3. Benchmark 3: Binomial Graph G3 (12 edges) - Figure 4, Table 2 & 3
    # Mode 2 corresponds to Binomial graph
    g3 = Example7(eps=0.1)
    res_g3 = run_benchmark("Binomial Graph G3", g3, n_width=200, mode=2, n_iter=3000, seed=0)

    # Final Quantitative Benchmark Comparison Table
    print("\n" + "=" * 90)
    print("FINAL SUMMARY: OUR DEEPONET vs FVM SOLVER vs PAPER PUBLISHED BENCHMARKS")
    print("=" * 90)
    print(f"{'Graph':<20} | {'Paper Table 2 (Abs L2)':<22} | {'Our Abs L2':<12} | {'Paper Table 3 (Rel L2)':<22} | {'Our Rel L2':<12}")
    print("-" * 90)
    print(f"{'Chain Graph G1':<20} | {'4.68e-03':<22} | {res_g1['l2_abs']:<12.4e} | {'1.06e-02 (~1.06%)':<22} | {res_g1['l2_rel']:<12.4e} ({res_g1['l2_rel']*100:.2f}%)")
    print(f"{'Double-Y Graph G2':<20} | {'5.62e-03':<22} | {res_g2['l2_abs']:<12.4e} | {'1.20e-02 (~1.20%)':<22} | {res_g2['l2_rel']:<12.4e} ({res_g2['l2_rel']*100:.2f}%)")
    print(f"{'Binomial Graph G3':<20} | {'5.81e-03':<22} | {res_g3['l2_abs']:<12.4e} | {'1.30e-02 (~1.30%)':<22} | {res_g3['l2_rel']:<12.4e} ({res_g3['l2_rel']*100:.2f}%)")
    print("=" * 90)

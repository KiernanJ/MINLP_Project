# %% [markdown]
# # Binary-First DiffOpt Surrogate for AC Unit Commitment (MINLP)
#
# **Baseline**: Constante-Flores & Li Algorithm 1 — Sequential MIQCAC
#   - Repeat until slack ≤ ε:
#       1. Form MIQCAC subproblem (4) linearised around (Vre,t, Vim,t), penalty ρt
#       2. Solve convex MIQCQP with binary u ∈ {0,1}  (Gurobi, no NonConvex=2)
#       3. Update Vre,t+1 ← vre,t+1,  Vim,t+1 ← vim,t+1
#       4. ρt+1 = min(μ·ρt, ρmax)   [EVERY iteration, unconditionally]
#   - Return the integer commitment from the last iteration
#
# **Proposed**: Binary-First DiffOpt Surrogate
#   - NN predicts binary u directly (Straight-Through Estimator, no continuous relaxation)
#   - Binary u is a FIXED PARAMETER in the differentiable NLP layer (QCAC physics)
#   - One forward pass → binary decision instantly; no solver loop at inference

# %% Cell 1: Setup & Grid Loading
import multiprocessing
import os, sys, warnings, time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from src.data_utils import parse_file_data, get_old_voltages
from src.model_binary import BinaryPredictor
from src.cvxpy_layer_binary import build_diffopt_nlp_layer
from src.parallel_worker_binary import run_binary_training_parallel
from src.formulation import (
    build_mp_ac_uc_rectangular, build_convex_ac_uc,
    build_single_ac_uc_rectangular, _precompute_bus_connectivity,
)

torch.manual_seed(42)
np.random.seed(42)

FILE_NAME  = 'case14_uctest'
FILE_PATH  = f'data/{FILE_NAME}.m'
data       = parse_file_data(FILE_PATH)
num_buses  = len(data.buses)
num_gens   = len(data.gens)
bus_ids    = sorted(data.buses.keys())
gen_ids    = sorted(data.gens.keys())
if __name__ == '__main__':
    print(f"Grid: {num_buses} buses, {num_gens} generators")

SCALE_MIN = 0.4
SCALE_MAX = 1.3
USE_PHASE_1 = True  # Toggle for Supervised Pre-Training

# %% Cell 2: Unlabeled Dataset (for self-supervised DiffOpt training)
os.makedirs('data', exist_ok=True)

def evaluate_fixed_u(data, u_fixed, pd_dict, qd_dict):
    """Fix binary u, re-solve true nonconvex AC-UC to get true cost."""
    from gurobipy import GRB
    m = build_single_ac_uc_rectangular(data)
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', 60.0)
    for b in bus_ids:
        m._pbal[b, 1].RHS = pd_dict.get(b, 0.0)
        m._qbal[b, 1].RHS = qd_dict.get(b, 0.0)
    for idx, g in enumerate(gen_ids):
        m._u[g, 1].lb = u_fixed[idx]
        m._u[g, 1].ub = u_fixed[idx]
    m.optimize()
    if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] or (
            m.status == GRB.TIME_LIMIT and m.SolCount > 0):
        pg = [m._pg[g, 1].X for g in gen_ids]
        return m.ObjVal, pg
    return np.nan, []

def run_miqcac_alg1(data, pd_dict, qd_dict, conn, vr_init, vi_init,
                    rho0=1000.0, mu=2.0, rho_max=1e5,
                    eps=1e-3, max_iter=20, mip_gap=0.01, time_limit=30.0):
    from gurobipy import GRB
    vr_t, vi_t, rho_t = vr_init.copy(), vi_init.copy(), rho0
    m_last = None
    for t in range(max_iter):
        m = build_convex_ac_uc(data, vr_t, vi_t, node_pd=pd_dict, node_qd=qd_dict,
                               conn=conn, penalty_weight=rho_t)
        m.setParam('OutputFlag', 0)
        m.setParam('MIPGap', mip_gap)
        m.setParam('TimeLimit', time_limit)
        m.optimize()
        m_last = m
        if m.status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL] and m.SolCount == 0:
            break
        vr_t = {b: m._vr[b, 1].X for b in bus_ids}
        vi_t = {b: m._vi[b, 1].X for b in bus_ids}
        rho_t = min(mu * rho_t, rho_max)
        slack_sum = 0.0
        for b in bus_ids:
            if hasattr(m, '_xi_c'): slack_sum += m._xi_c[b, 1].X
        for br_key in m._xij_c: slack_sum += m._xij_c[br_key].X
        for br_key in m._xij_s: slack_sum += m._xij_s[br_key].X
        if slack_sum <= eps:
            break
    if m_last is not None and m_last.SolCount > 0:
        return [int(round(m_last._u[g, 1].X)) for g in gen_ids], True
    return None, False
unsup_path = f'data/unsupervised_data_{FILE_NAME}.pt'

if os.path.exists(unsup_path):
    if __name__ == '__main__': print("Loading cached unlabeled dataset...")
    ud = torch.load(unsup_path, weights_only=False)
    X_unsup_t = ud['X_unsup_t']
else:
    if __name__ == '__main__': print("Generating unlabeled dataset...")
    n_unlabeled   = 200
    X_unsup       = []
    for _ in range(n_unlabeled):
        sc = np.random.uniform(SCALE_MIN, SCALE_MAX)
        t_pd, t_qd, t_inp = {}, {}, []
        for l_id, load in data.loads.items():
            b = load['load_bus']
            t_pd[b] = load['pd'] * sc * np.random.normal(1.0, 0.05)
            t_qd[b] = load['qd'] * sc * np.random.normal(1.0, 0.05)
        for b in bus_ids:
            t_inp.extend([t_pd.get(b, 0.0), t_qd.get(b, 0.0)])
        X_unsup.append(t_inp)
    X_unsup_t = torch.tensor(np.array(X_unsup), dtype=torch.float32)
    torch.save({'X_unsup_t': X_unsup_t}, unsup_path)

if __name__ == '__main__': print(f"Unlabeled samples: {len(X_unsup_t)}")

# %% Cell 2.5: Phase 1 Supervised Data Generation
sup_path = f'data/supervised_data_{FILE_NAME}.pt'
n_supervised = 20

if USE_PHASE_1:
    if os.path.exists(sup_path):
        if __name__ == '__main__': print("\nLoading cached supervised dataset for Phase 1...")
        sup_data = torch.load(sup_path, weights_only=False)
    else:
        if __name__ == '__main__': print(f"\nGenerating {n_supervised} true MINLP solutions for Phase 1...")
        sup_data = []
        if __name__ == '__main__':
            from gurobipy import GRB
            while len(sup_data) < n_supervised:
                sc = np.random.uniform(SCALE_MIN, SCALE_MAX)
                t_pd, t_qd, t_inp = {}, {}, []
                for l_id, load in data.loads.items():
                    b = load['load_bus']
                    t_pd[b] = load['pd'] * sc * np.random.normal(1.0, 0.05)
                    t_qd[b] = load['qd'] * sc * np.random.normal(1.0, 0.05)
                for b in bus_ids:
                    t_inp.extend([t_pd.get(b, 0.0), t_qd.get(b, 0.0)])
                
                m_true = build_single_ac_uc_rectangular(data)
                m_true.setParam('OutputFlag', 0)
                m_true.setParam('TimeLimit', 60.0)
                for b in bus_ids:
                    m_true._pbal[b, 1].RHS = t_pd.get(b, 0.0)
                    m_true._qbal[b, 1].RHS = t_qd.get(b, 0.0)
                m_true.optimize()
                if m_true.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                    sup_data.append({
                        'x_tensor': torch.tensor(t_inp, dtype=torch.float32),
                        'u_true': [int(round(m_true._u[g, 1].X)) for g in gen_ids]
                    })
                    print(f"  {len(sup_data)}/{n_supervised}", end="\r")
            torch.save(sup_data, sup_path)
            print("\nDone generating supervised data!")

# %% Cell 3: Model & NLP Differentiable Layer
model = BinaryPredictor(num_buses, num_gens)

if __name__ == '__main__' and USE_PHASE_1:
    print("\n--- Phase 1: Supervised Pre-Training ---")
    opt = optim.Adam(model.parameters(), lr=5e-3)
    bce = nn.BCELoss()
    
    if isinstance(sup_data, dict):
        X_sup = sup_data['X_sup_t']
        Y_sup = sup_data['Y_u_sup_t']
    else:
        X_sup = torch.stack([inst['x_tensor'] for inst in sup_data])
        Y_sup = torch.tensor([inst['u_true'] for inst in sup_data], dtype=torch.float32)
    
    for ep in range(1500):
        model.train()
        opt.zero_grad()
        _, u_prob, _, _, _ = model(X_sup)
        loss = bce(u_prob, Y_sup)
        loss.backward()
        opt.step()
        if (ep + 1) % 250 == 0:
            print(f"Phase 1 | Epoch {ep+1:>4}/1500 | BCE Loss: {loss.item():.4f}")
elif __name__ == '__main__':
    print("\n--- Phase 1: SKIPPED (Pure Unsupervised DiffOpt) ---")
    
    print("\nBinaryPredictor model:")
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nBuilding differentiable NLP layer (u fixed by NN)...")
    nlp_layer, b_idx, g_idx = build_diffopt_nlp_layer(data)

# %% Cell 4: Self-Supervised DiffOpt Training (Parallel FedAvg)
# No Phase 1 supervised pre-training — the NLP layer itself is the teacher.
# Loss = generation cost (flows through STE to binary u) + QCAC slack penalty.
# Workers run in parallel (same FedAvg pattern as DF_model_unsupervised.ipynb).
if __name__ == '__main__':
    print("\n--- Self-Supervised DiffOpt Training (Binary u → NLP Layer, Parallel) ---")
    
    EPOCHS      = 10
    BATCH_SIZE  = 50
    N_WORKERS   = 12      # match your CPU core count
    SLACK_W     = 5000.0
    LR          = 5e-4
    SOLVER_EPS  = 1e-3
    SOLVER_ITERS= 8000
    
    model, history = run_binary_training_parallel(
        model        = model,
        X_unsup_t    = X_unsup_t,
        data_file_path = os.path.abspath(FILE_PATH),
        num_buses    = num_buses,
        num_gens     = num_gens,
        epochs       = EPOCHS,
        batch_size   = BATCH_SIZE,
        n_workers    = N_WORKERS,
        slack_weight = SLACK_W,
        solver_eps   = SOLVER_EPS,
        solver_iters = SOLVER_ITERS,
        lr           = LR,
        use_phase_1  = USE_PHASE_1
    )
    print("Training Complete!")
    
    # %% Cell 5: Convergence Plot
    plt.figure(figsize=(8, 4))
    plt.plot(history, marker='o', markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("Avg Loss")
    plt.title("DiffOpt Training Loss"); plt.grid(True, ls='--', alpha=0.6)
    plt.tight_layout(); plt.show()
    
    # %% Cell 6: Inference Test
    print("\n--- Inference Test ---")
    model.eval()
    np.random.seed(142)
    test_scale = np.random.uniform(SCALE_MIN, SCALE_MAX)
    test_pd, test_qd, test_inp = {}, {}, []
    for l_id, load in data.loads.items():
        b = load['load_bus']
        test_pd[b] = load['pd'] * test_scale * np.random.normal(1.0, 0.05)
        test_qd[b] = load['qd'] * test_scale * np.random.normal(1.0, 0.05)
    for b in bus_ids:
        test_inp.extend([test_pd.get(b, 0.0), test_qd.get(b, 0.0)])
    
    x_test = torch.tensor(test_inp, dtype=torch.float32).unsqueeze(0)
    pd_t   = x_test[:, :num_buses]
    qd_t   = x_test[:, num_buses:]
    
    t0 = time.time()
    with torch.no_grad():
        u_bin, u_prob, vr_base, vi_base, rho = model(x_test)
        print(f"u_prob  = {u_prob.squeeze().numpy().round(3)}")
        print(f"u_binary= {u_bin.squeeze().numpy().astype(int)}")
        try:
            pg, qg, vr, vi, xi_c, xij_c, xij_s = nlp_layer(
                u_bin, vr_base, vi_base, rho.squeeze(-1), pd_t, qd_t,
                solver_args={'max_iters': 10000, 'eps': 1e-3})
            slack = (xi_c.sum() + xij_c.sum() + xij_s.sum()).item()
            print(f"pg      = {pg.squeeze().numpy().round(4)}")
            print(f"Slack   = {slack:.6f}")
        except Exception as e:
            print(f"NLP layer failed: {e}")
    print(f"Inference time: {time.time()-t0:.3f}s")
    
    # %% Cell 7: Benchmark Setup
    print("\n--- Benchmark: Can Li (Sequential MIQCAC) vs Binary-First DiffOpt ---")
    from gurobipy import GRB
    
    NUM_TEST       = 20
    BENCHMARK_PATH = f'data/test_benchmark{FILE_NAME}.pt'
    

    
    # %% Cell 8: Load / Generate Ground Truth
    if os.path.exists(BENCHMARK_PATH):
        print("Loading cached ground truth...")
        test_instances = torch.load(BENCHMARK_PATH, weights_only=False)
    else:
        print(f"Generating {NUM_TEST} true MINLP solutions...")
        test_instances = []
        while len(test_instances) < NUM_TEST:
            sc = np.random.uniform(SCALE_MIN, SCALE_MAX)
            t_pd, t_qd, t_inp = {}, {}, []
            for l_id, load in data.loads.items():
                b = load['load_bus']
                t_pd[b] = load['pd'] * sc * np.random.normal(1.0, 0.05)
                t_qd[b] = load['qd'] * sc * np.random.normal(1.0, 0.05)
            for b in bus_ids:
                t_inp.extend([t_pd.get(b, 0.0), t_qd.get(b, 0.0)])
            t0 = time.time()
            m_true = build_single_ac_uc_rectangular(data)
            m_true.setParam('OutputFlag', 0)
            m_true.setParam('TimeLimit', 60.0)
            for b in bus_ids:
                m_true._pbal[b, 1].RHS = t_pd.get(b, 0.0)
                m_true._qbal[b, 1].RHS = t_qd.get(b, 0.0)
            m_true.optimize()
            if m_true.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                inst = {
                    'test_pd': t_pd, 'test_qd': t_qd, 'x_tensor': t_inp,
                    'cost_true': m_true.ObjVal,
                    'u_true':  [int(round(m_true._u[g, 1].X)) for g in gen_ids],
                    'pg_true': [m_true._pg[g, 1].X for g in gen_ids],
                    'time_true': time.time() - t0,
                }
                
                # Warm-start voltages for Can Li
                m_base = build_mp_ac_uc_rectangular(data, demand_curve=[1.0])
                m_base.setParam('OutputFlag', 0)
                m_base.optimize()
                vr_init_g, vi_init_g = get_old_voltages(m_base, data)
                conn = _precompute_bus_connectivity(data)
                
                t_start = time.time()
                u_cl, ok = run_miqcac_alg1(
                    data, t_pd, t_qd, conn, vr_init_g, vi_init_g,
                    rho0=1000.0, mu=2.0, rho_max=1e5, eps=1e-3, max_iter=20,
                    mip_gap=0.01, time_limit=30.0,
                )
                t_cl = time.time() - t_start
                
                if ok and u_cl is not None:
                    cost_cl, pg_cl = evaluate_fixed_u(data, u_cl, t_pd, t_qd)
                    inst.update({
                        'u_cl': u_cl, 'pg_cl': pg_cl, 'cost_cl': cost_cl, 'time_cl': t_cl
                    })
                else:
                    inst.update({'u_cl': None, 'pg_cl': None, 'cost_cl': None, 'time_cl': None})

                test_instances.append(inst)
                print(f"  {len(test_instances)}/{NUM_TEST}", end="\r")
        torch.save(test_instances, BENCHMARK_PATH)
        print("\nDone!")
    
    # Warm-start voltages (flat start ≈ 1.0 + 0.0j pu)
    m_base = build_mp_ac_uc_rectangular(data, demand_curve=[1.0])
    m_base.setParam('OutputFlag', 0)
    m_base.optimize()
    vr_init_global, vi_init_global = get_old_voltages(m_base, data)
    conn = _precompute_bus_connectivity(data)
    
    # %% Cell 9: Evaluate Both Methods
    results = {
        'canli':  {'err_d': [], 'err_c': [], 'gap': [], 'time': []},
        'binary': {'err_d': [], 'err_c': [], 'gap': [], 'time': []},
        'time_true': [inst['time_true'] for inst in test_instances[:NUM_TEST]],
    }
    
    model.eval()
    
    for i, inst in enumerate(test_instances[:NUM_TEST]):
        print(f"Evaluating {i+1}/{NUM_TEST}...", end="\r")
        t_pd, t_qd  = inst['test_pd'], inst['test_qd']
        cost_true   = inst['cost_true']
        u_true      = inst['u_true']
        pg_true     = inst['pg_true']
        x_t         = torch.tensor(inst['x_tensor'], dtype=torch.float32).unsqueeze(0)
    
        # ── A. Can Li Algorithm 1: Sequential MIQCAC ─────────────────────────────
        # Now we just load the cached Can Li results!
        u_cl = inst.get('u_cl', None)
        if u_cl is not None and inst.get('cost_cl', None) is not None:
            cost_cl, pg_cl, t_cl = inst['cost_cl'], inst['pg_cl'], inst['time_cl']
            if not np.isnan(cost_cl):
                results['canli']['time'].append(t_cl)
                results['canli']['err_d'].append(
                    np.sum(np.abs(np.array(u_true) - np.array(u_cl))) / num_gens * 100)
                results['canli']['err_c'].append(
                    np.mean(np.abs(np.array(pg_true) - np.array(pg_cl))) * 100)
                results['canli']['gap'].append(
                    abs(cost_cl - cost_true) / max(cost_true, 1e-5) * 100)
        
        # ── B. Binary-First DiffOpt Surrogate ────────────────────────────────────
        t_start = time.time()
        with torch.no_grad():
            u_bin, _, _, _, _ = model(x_t)
            u_do = u_bin.squeeze(0).numpy().astype(int).tolist()
        t_do = time.time() - t_start
    
        cost_do, pg_do = evaluate_fixed_u(data, u_do, t_pd, t_qd)
        if not np.isnan(cost_do):
            results['binary']['time'].append(t_do)
            results['binary']['err_d'].append(
                np.sum(np.abs(np.array(u_true) - np.array(u_do))) / num_gens * 100)
            results['binary']['err_c'].append(
                np.mean(np.abs(np.array(pg_true) - np.array(pg_do))) * 100)
            results['binary']['gap'].append(
                abs(cost_do - cost_true) / max(cost_true, 1e-5) * 100)
    
        print(f"  [{i+1:>2}] u_true={u_true}  CanLi={u_cl}  DiffOpt={u_do}  "
              f"gap_CL={results['canli']['gap'][-1] if results['canli']['gap'] else float('nan'):.1f}%  "
              f"gap_DO={results['binary']['gap'][-1] if results['binary']['gap'] else float('nan'):.1f}%")
    
    # %% Cell 10: Plots
    print("\n\nGenerating benchmark plots...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    labels = ['Can Li Alg 1\n(Seq. MIQCAC)', 'Binary DiffOpt\n(Proposed)']
    
    def fmt(ax, title, ylabel):
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, ls='--', alpha=0.6)
    
    axes[0].boxplot([results['canli']['err_d'], results['binary']['err_d']], labels=labels)
    fmt(axes[0], 'Discrete Decision Error (%)', '% Error')
    
    axes[1].boxplot([results['canli']['err_c'], results['binary']['err_c']], labels=labels)
    fmt(axes[1], 'Continuous Decision Error (%)', '% MAE')
    
    axes[2].boxplot([results['canli']['gap'], results['binary']['gap']], labels=labels)
    fmt(axes[2], 'Optimality Gap (%)', '% Gap')
    
    axes[3].step(np.sort(results['time_true']),
                 np.linspace(0, 100, len(results['time_true'])),
                 color='darkred', lw=2.5, label='True MINLP')
    if results['canli']['time']:
        axes[3].step(np.sort(results['canli']['time']),
                     np.linspace(0, 100, len(results['canli']['time'])),
                     color='navy', lw=2.5, label='Can Li Alg 1 (Seq. MIQCAC)')
    if results['binary']['time']:
        axes[3].step(np.sort(results['binary']['time']),
                     np.linspace(0, 100, len(results['binary']['time'])),
                     color='forestgreen', ls='--', lw=2.5, label='Binary DiffOpt (Proposed)')
    axes[3].set_xscale('log')
    axes[3].set_title('Computational Time CDF', fontsize=13)
    axes[3].set_xlabel('Time (s, log scale)', fontsize=11)
    axes[3].set_ylabel('% of Instances Solved', fontsize=11)
    axes[3].legend(loc='lower right', fontsize=10)
    axes[3].grid(True, ls='--', alpha=0.6)
    
    plt.suptitle('Binary-First DiffOpt vs Can Li Alg 1 (Seq. MIQCAC) vs True MINLP',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('data/benchmark_binary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # %% Cell 11: Summary Table
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"{'Metric':<30} {'Can Li Alg 1':<18} {'Binary DiffOpt':<18}")
    print("-" * 68)
    for name, key in [("Discrete Err % (mean)",  "err_d"),
                       ("Continuous Err % (mean)", "err_c"),
                       ("Optimality Gap % (mean)", "gap"),
                       ("Solve Time s (mean)",      "time")]:
        cl = np.mean(results['canli'][key])  if results['canli'][key]  else float('nan')
        do = np.mean(results['binary'][key]) if results['binary'][key] else float('nan')
        print(f"{name:<30} {cl:>16.3f} {do:>16.3f}")
    print(f"{'True MINLP Time s (mean)':<30} {np.mean(results['time_true']):>16.3f}")
    
    # %%

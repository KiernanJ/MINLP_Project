"""
src/parallel_worker_binary.py
FedAvg parallel worker for the Binary-First DiffOpt training loop.

Each worker receives the current BinaryPredictor weights, runs sequential
per-sample forward → NLP-layer → backward → step on its chunk of unlabeled
samples, then returns its final updated weights.

Main process averages weights across workers (FedAvg), same as parallel_worker.py.
"""

import os
import sys
import io
import warnings
import time
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# WORKER  (top-level so ProcessPoolExecutor can pickle it)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_binary(args):
    """
    FedAvg worker for BinaryPredictor + NLP layer.
    Runs sequential per-sample SGD on its chunk, returns final weights.
    """
    (project_root, data_file_path,
     model_state_np, samples_np,
     num_buses, num_gens,
     slack_weight, solver_eps, solver_iters, lr, use_phase_1) = args

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings('ignore')

    from src.model_binary      import BinaryPredictor
    from src.data_utils        import parse_file_data
    from src.cvxpy_layer_binary import build_diffopt_nlp_layer
    from contextlib import redirect_stdout

    # Rebuild model + load weights
    data  = parse_file_data(data_file_path)
    model = BinaryPredictor(num_buses, num_gens)
    state = {k: torch.tensor(v, dtype=torch.float32)
             for k, v in model_state_np.items()}
    model.load_state_dict(state)
    model.train()

    gen_ids = sorted(data.gens.keys())
    g_idx   = {g: i for i, g in enumerate(gen_ids)}

    if use_phase_1:
        # Phase 1 is used: Freeze BOTH the trunk and head_u to strictly retain Phase 1 knowledge.
        local_opt = torch.optim.Adam([
            {'params': model.head_v.parameters(), 'lr': lr},      # train voltages
            {'params': model.head_rho.parameters(), 'lr': lr}     # train rho
        ])
    else:
        # Phase 1 is NOT used: Unsupervised training must optimize ALL layers.
        local_opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Build NLP layer silently
    with redirect_stdout(io.StringIO()):
        nlp_layer, b_idx, g_idx_map = build_diffopt_nlp_layer(data)

    total_loss, n_valid = 0.0, 0

    for sample_np in samples_np:
        x_in = torch.tensor(sample_np, dtype=torch.float32).unsqueeze(0)
        pd_t = x_in[:, :num_buses]
        qd_t = x_in[:, num_buses:]

        local_opt.zero_grad()
        try:
            u_bin, u_prob, vr_base, vi_base, rho = model(x_in)

            if use_phase_1:
                # Pass the HARD binary prediction (u_bin) into the physics solver.
                # We don't need continuous gradients for u anymore since we froze it.
                pg, qg, vr, vi, xi_c, xij_c, xij_s = nlp_layer(
                    u_bin, vr_base, vi_base, rho.squeeze(-1), pd_t, qd_t,
                    solver_args={'max_iters': solver_iters, 'eps': solver_eps},
                )
                # Generation cost
                gen_cost = sum(
                    data.gens[g]['cost'][0] * pg[0, g_idx_map[g]]**2
                    + data.gens[g]['cost'][1] * pg[0, g_idx_map[g]]
                    + data.gens[g]['cost'][2] * u_bin[0, g_idx_map[g]]
                    for g in gen_ids
                )
            else:
                # Pass continuous u_prob so the NLP KKT conditions pass gradients back to the NN.
                pg, qg, vr, vi, xi_c, xij_c, xij_s = nlp_layer(
                    u_prob, vr_base, vi_base, rho.squeeze(-1), pd_t, qd_t,
                    solver_args={'max_iters': solver_iters, 'eps': solver_eps},
                )
                # Generation cost must be linked to u_prob directly
                gen_cost = sum(
                    data.gens[g]['cost'][0] * pg[0, g_idx_map[g]]**2
                    + data.gens[g]['cost'][1] * pg[0, g_idx_map[g]]
                    + data.gens[g]['cost'][2] * u_prob[0, g_idx_map[g]]
                    for g in gen_ids
                )
            slack = torch.sum(xi_c) + torch.sum(xij_c) + torch.sum(xij_s)
            
            if not use_phase_1:
                # ── ROBUST CAPACITY PENALTY (Unsupervised Fix) ──
                # We must enforce a safety margin (e.g., 20%) because the AC grid has physical power losses!
                # If capacity exactly equals demand, the MINLP solver will fail due to line losses.
                total_demand_p = torch.sum(pd_t)
                total_cap_p    = sum(data.gens[g]['pmax'] * u_prob[0, g_idx_map[g]] for g in gen_ids)
                pen_p          = 50000.0 * torch.nn.functional.relu((total_demand_p * 1.2) - total_cap_p)

                total_demand_q = torch.sum(qd_t)
                total_cap_q    = sum(data.gens[g]['qmax'] * u_prob[0, g_idx_map[g]] for g in gen_ids)
                pen_q          = 50000.0 * torch.nn.functional.relu((total_demand_q * 1.2) - total_cap_q)

                capacity_penalty = pen_p + pen_q
                loss = gen_cost + slack_weight * slack + capacity_penalty
            else:
                loss  = gen_cost + slack_weight * slack

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            local_opt.step()

            total_loss += loss.item()
            n_valid    += 1

        except Exception:
            pass

    final_weights = {k: v.detach().numpy().copy()
                     for k, v in model.state_dict().items()}
    return final_weights, total_loss, n_valid


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_binary_training_parallel(
    model,
    X_unsup_t,
    data_file_path,
    num_buses,
    num_gens,
    epochs       = 30,
    batch_size   = 50,
    n_workers    = 6,
    slack_weight = 5000.0,
    solver_eps   = 1e-3,
    solver_iters = 8000,
    lr           = 5e-4,
    use_phase_1  = True,
):
    """
    Parallel FedAvg training for BinaryPredictor.

    Drop-in replacement for the sequential training loop in DF_model_binary.py.
    Returns the trained model and per-epoch average loss history.
    """
    project_root = os.path.abspath('.')
    model.train()
    history = []

    print(f"Binary DiffOpt Training (FedAvg): {n_workers} workers | "
          f"{batch_size} samples/epoch | eps={solver_eps} | lr={lr}")

    for epoch in range(epochs):
        t0 = time.time()

        # Random mini-batch
        idx_batch  = torch.randperm(len(X_unsup_t))[:batch_size]
        samples_np = X_unsup_t[idx_batch].numpy()

        # Snapshot current weights (numpy, picklable)
        model_state_np = {k: v.detach().numpy()
                          for k, v in model.state_dict().items()}

        # Split into worker chunks
        chunk_size = max(1, -(-batch_size // n_workers))   # ceiling div
        chunks     = [samples_np[i: i + chunk_size]
                      for i in range(0, len(samples_np), chunk_size)]

        args_list = [
            (project_root, data_file_path,
             model_state_np, chunk,
             num_buses, num_gens,
             slack_weight, solver_eps, solver_iters, lr, use_phase_1)
            for chunk in chunks
        ]

        # Run workers in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_worker_binary, args_list))

        total_valid    = sum(r[2] for r in results)
        total_loss_sum = sum(r[1] for r in results)

        if total_valid > 0:
            # FedAvg: weighted average of worker weights
            avg_state = {}
            for key in model_state_np.keys():
                weighted_sum, total_w = None, 0
                for r in results:
                    if r[2] > 0:
                        contrib      = r[0][key] * r[2]
                        weighted_sum = contrib if weighted_sum is None \
                                       else weighted_sum + contrib
                        total_w     += r[2]
                if weighted_sum is not None:
                    avg_state[key] = torch.tensor(
                        weighted_sum / total_w, dtype=torch.float32)
            model.load_state_dict(avg_state)

        avg_loss = total_loss_sum / max(total_valid, 1)
        history.append(avg_loss)
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Solved {total_valid}/{batch_size} | "
              f"Avg Loss: {avg_loss:.2f} | "
              f"Time: {time.time()-t0:.1f}s")

    print("Binary DiffOpt Training Complete!")
    return model, history

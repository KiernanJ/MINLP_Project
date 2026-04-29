"""
src/cvxpy_layer_binary.py
Differentiable NLP layer where u is a PARAMETER (fixed by the NN),
not a variable. The layer solves only for continuous physics (pg, qg, vr, vi).
"""
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np


def build_diffopt_nlp_layer(data):
    """
    Build a differentiable CVXPY layer for the AC-OPF with FIXED binary u.
    u is injected as a parameter from the NN prediction.
    Returns: (CvxpyLayer, b_idx, g_idx)
    """
    from .data_utils import calc_branch_y, calc_branch_t

    print("Building Differentiable NLP Layer (u fixed by NN)...")

    bus_ids = sorted(list(data.buses.keys()))
    gen_ids = sorted(list(data.gens.keys()))
    branch_ids = sorted(list(data.branches.keys()))
    b_idx = {b: i for i, b in enumerate(bus_ids)}
    g_idx = {g: i for i, g in enumerate(gen_ids)}
    br_idx = {br: i for i, br in enumerate(branch_ids)}
    N_b, N_g, N_br = len(bus_ids), len(gen_ids), len(branch_ids)

    # --- Variables (continuous only) ---
    vr = cp.Variable(N_b)
    vi = cp.Variable(N_b)
    c_ii = cp.Variable(N_b)
    c_ij = cp.Variable(N_br)
    s_ij = cp.Variable(N_br)
    pg = cp.Variable(N_g)
    qg = cp.Variable(N_g)
    p_fr = cp.Variable(N_br)
    q_fr = cp.Variable(N_br)
    p_to = cp.Variable(N_br)
    q_to = cp.Variable(N_br)
    xi_c = cp.Variable(N_b, nonneg=True)
    xij_c = cp.Variable(N_br, nonneg=True)
    xij_s = cp.Variable(N_br, nonneg=True)

    # --- Parameters (from NN + environment) ---
    u_fixed = cp.Parameter(N_g, nonneg=True)   # binary u from NN
    vr_base = cp.Parameter(N_b)
    vi_base = cp.Parameter(N_b)
    rho = cp.Parameter(nonneg=True)
    pd = cp.Parameter(N_b)
    qd = cp.Parameter(N_b)

    constraints = []

    # Reference bus
    for i in [b_idx[k] for k, v in data.buses.items() if v['bus_type'] == 3]:
        constraints += [vi[i] == 0.0, vr[i] >= 0.0]

    # Generator bounds (linked to u_fixed parameter)
    for g_id, gen in data.gens.items():
        g = g_idx[g_id]
        constraints += [
            pg[g] >= gen['pmin'] * u_fixed[g],
            pg[g] <= gen['pmax'] * u_fixed[g],
            qg[g] >= gen['qmin'] * u_fixed[g],
            qg[g] <= gen['qmax'] * u_fixed[g],
        ]

    # Nodal QCAC constraints
    for b_id, bus in data.buses.items():
        i = b_idx[b_id]
        constraints += [
            c_ii[i] >= bus['vmin']**2,
            c_ii[i] <= bus['vmax']**2,
            c_ii[i] >= cp.square(vr[i]) + cp.square(vi[i]),
        ]
        lin = 2.0 * (cp.multiply(vr_base[i], vr[i]) + cp.multiply(vi_base[i], vi[i]))
        cst = cp.square(vr_base[i]) + cp.square(vi_base[i])
        constraints.append(c_ii[i] <= lin - cst + xi_c[i])

    # Branch QCAC constraints & physics
    for br_id, branch in data.branches.items():
        k = br_idx[br_id]
        i = b_idx[str(branch['f_bus'])]
        j = b_idx[str(branch['t_bus'])]
        g, b = calc_branch_y(branch)
        tr, ti_ = calc_branch_t(branch)
        g_fr, b_fr = branch['g_fr'], branch['b_fr']
        g_to, b_to = branch['g_to'], branch['b_to']
        tm2 = branch['tap'] ** 2
        ra = branch['rate_a']

        # 4d
        rhs_4d = (xij_c[k]
                  + 2.0*cp.multiply(vr[i]-vr[j], vr_base[i]-vr_base[j])
                  + 2.0*cp.multiply(vi[i]-vi[j], vi_base[i]-vi_base[j])
                  - (cp.square(vr_base[i]-vr_base[j]) + cp.square(vi_base[i]-vi_base[j])))
        constraints.append(cp.square(vr[i]+vr[j]) + cp.square(vi[i]+vi[j]) - 4.0*c_ij[k] <= rhs_4d)
        # 4e
        rhs_4e = (xij_c[k]
                  + 2.0*cp.multiply(vr[i]+vr[j], vr_base[i]+vr_base[j])
                  + 2.0*cp.multiply(vi[i]+vi[j], vi_base[i]+vi_base[j])
                  - (cp.square(vr_base[i]+vr_base[j]) + cp.square(vi_base[i]+vi_base[j])))
        constraints.append(cp.square(vr[i]-vr[j]) + cp.square(vi[i]-vi[j]) + 4.0*c_ij[k] <= rhs_4e)
        # 4f1
        rhs_4f1 = (xij_s[k]
                   + 2.0*cp.multiply(vr[i]+vi[j], vr_base[i]+vi_base[j])
                   + 2.0*cp.multiply(vr[j]-vi[i], vr_base[j]-vi_base[i])
                   - (cp.square(vr_base[i]+vi_base[j]) + cp.square(vr_base[j]-vi_base[i])))
        constraints.append(cp.square(vr[i]-vi[j]) + cp.square(vr[j]+vi[i]) + 4.0*s_ij[k] <= rhs_4f1)
        # 4f2
        rhs_4f2 = (xij_s[k]
                   + 2.0*cp.multiply(vr[i]-vi[j], vr_base[i]-vi_base[j])
                   + 2.0*cp.multiply(vr[j]+vi[i], vr_base[j]+vi_base[i])
                   - (cp.square(vr_base[i]-vi_base[j]) + cp.square(vr_base[j]+vi_base[i])))
        constraints.append(cp.square(vr[i]+vi[j]) + cp.square(vr[j]-vi[i]) - 4.0*s_ij[k] <= rhs_4f2)

        # Power flow
        constraints += [
            p_fr[k] == (g+g_fr)/tm2*c_ii[i] + (-g*tr+b*ti_)/tm2*c_ij[k] + (-b*tr-g*ti_)/tm2*s_ij[k],
            q_fr[k] == -(b+b_fr)/tm2*c_ii[i] - (-b*tr-g*ti_)/tm2*c_ij[k] + (-g*tr+b*ti_)/tm2*s_ij[k],
            p_to[k] == (g+g_to)*c_ii[j] + (-g*tr-b*ti_)/tm2*c_ij[k] + (-b*tr+g*ti_)/tm2*s_ij[k],
            q_to[k] == -(b+b_to)*c_ii[j] - (-b*tr+g*ti_)/tm2*c_ij[k] + (-g*tr-b*ti_)/tm2*s_ij[k],
            cp.square(p_fr[k]) + cp.square(q_fr[k]) <= ra**2,
            cp.square(p_to[k]) + cp.square(q_to[k]) <= ra**2,
        ]

    # Nodal power balance
    for b_id, bus in data.buses.items():
        i = b_idx[b_id]
        bus_gens = [g_idx[gg] for gg, d in data.gens.items() if d['gen_bus'] == b_id]
        br_fr = [br_idx[kk] for kk, bb in data.branches.items() if str(bb['f_bus']) == b_id]
        br_to = [br_idx[kk] for kk, bb in data.branches.items() if str(bb['t_bus']) == b_id]
        gs = sum(s['gs'] for s in data.shunts.values() if s['shunt_bus'] == b_id)
        bs = sum(s['bs'] for s in data.shunts.values() if s['shunt_bus'] == b_id)
        pg_s = cp.sum(pg[bus_gens]) if bus_gens else 0.0
        qg_s = cp.sum(qg[bus_gens]) if bus_gens else 0.0
        pf_s = cp.sum(p_fr[br_fr]) if br_fr else 0.0
        pt_s = cp.sum(p_to[br_to]) if br_to else 0.0
        qf_s = cp.sum(q_fr[br_fr]) if br_fr else 0.0
        qt_s = cp.sum(q_to[br_to]) if br_to else 0.0
        constraints.append(pg_s - gs*c_ii[i] - pf_s - pt_s == pd[i])
        constraints.append(qg_s + bs*c_ii[i] - qf_s - qt_s == qd[i])

    # Objective: generation cost + slack penalty
    cost_expr = 0
    for g_id, gen in data.gens.items():
        gg = g_idx[g_id]
        c2, c1, c0 = gen['cost']
        cost_expr += c2*cp.square(pg[gg]) + c1*pg[gg] + c0*u_fixed[gg]

    slack_sum = cp.sum(xi_c) + cp.sum(xij_c) + cp.sum(xij_s)
    obj = cp.Minimize(cost_expr + rho * slack_sum)

    prob = cp.Problem(obj, constraints)
    assert prob.is_dcp(), "Not DCP!"
    print(f"NLP Layer compiled. DCP: TRUE")

    layer = CvxpyLayer(
        prob,
        parameters=[u_fixed, vr_base, vi_base, rho, pd, qd],
        variables=[pg, qg, vr, vi, xi_c, xij_c, xij_s],
    )
    return layer, b_idx, g_idx

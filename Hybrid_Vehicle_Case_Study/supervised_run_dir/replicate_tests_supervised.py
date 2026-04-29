# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time
import warnings
from collections import defaultdict
from tqdm import tqdm
import json
import os

warnings.filterwarnings("ignore", message="Solved/Inaccurate")

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    if isinstance(obj, tuple):
        return str(obj)
    return obj

# ==============================================================================
# 0. HYPERPARAMETERS & PHYSICS SETUP
# ==============================================================================
T = 30                  
N_TRAIN_MAX = 100            
N_TEST = 100             
TRAINING_EPOCHS = 30
N_INSTANCES = 1

np.random.seed(42)
torch.manual_seed(42)

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

# Functions parameterized by `veh` (vehicle parameters dictionary)

def build_gurobi_miqcqp(D_profile, veh, is_fixed_z=False, z_fixed=None):
    m = gp.Model(env=env)
    m.Params.NonConvex = 2 
    m.Params.TimeLimit = 60.0
    
    E = m.addVars(T+1, lb=0, ub=veh['E_max'], vtype=GRB.CONTINUOUS)
    P_eng = m.addVars(T, lb=0, ub=veh['P_max'], vtype=GRB.CONTINUOUS)
    P_batt = m.addVars(T, lb=-veh['P_batt_max'], ub=veh['P_batt_max'], vtype=GRB.CONTINUOUS)
    
    if is_fixed_z:
        z = {t: z_fixed[t] for t in range(T)}
    else:
        z = m.addVars(T, lb=0, ub=veh['S_MODES'], vtype=GRB.INTEGER)
    
    m.addConstr(E[0] == veh['E_init'])
    
    for t in range(T):
        m.addConstr(E[t+1] <= E[t] - veh['tau'] * P_batt[t] - veh['gamma'] * P_batt[t]*P_batt[t])
        m.addConstr(P_eng[t] <= z[t] * (veh['P_max'] / veh['S_MODES']))
        m.addConstr(P_batt[t] + P_eng[t] >= D_profile[t])

    obj = gp.quicksum(veh['alpha'][t] * P_eng[t]*P_eng[t] + veh['beta'][t] * z[t] for t in range(T)) + veh['eta'] * (veh['E_max'] - E[T])
    m.setObjective(obj, GRB.MINIMIZE)
    return m, z, E, P_eng, P_batt

def solve_true_miqcqp_full(D_profile, veh):
    m, z, E, P_eng, P_batt = build_gurobi_miqcqp(D_profile, veh)
    m.optimize()
    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        if m.SolCount > 0:
            return (
                np.array([z[t].X for t in range(T)]),
                np.array([E[t].X for t in range(T+1)]),
                np.array([P_eng[t].X for t in range(T)]),
                np.array([P_batt[t].X for t in range(T)]),
                m.ObjVal
            )
    return np.zeros(T), np.zeros(T+1), np.zeros(T), np.zeros(T), float('inf')

def evaluate_fixed_z_miqcqp(D_profile, veh, z_fixed):
    m, _, E, P_eng, P_batt = build_gurobi_miqcqp(D_profile, veh, is_fixed_z=True, z_fixed=z_fixed)
    m.optimize()
    if m.status == GRB.OPTIMAL:
        E_opt = np.array([E[t].X for t in range(T+1)])
        P_eng_opt = np.array([P_eng[t].X for t in range(T)])
        P_batt_opt = np.array([P_batt[t].X for t in range(T)])
        return m.ObjVal, E_opt, P_eng_opt, P_batt_opt
    return float('inf'), np.zeros(T+1), np.zeros(T), np.zeros(T)

def qi_zhang_miqcqp_restoration(D_profile, veh, z_frac):
    m = gp.Model(env=env)
    m.Params.NonConvex = 2
    m.Params.TimeLimit = 60.0
    
    E = m.addVars(T+1, lb=0, ub=veh['E_max'], vtype=GRB.CONTINUOUS)
    P_eng = m.addVars(T, lb=0, ub=veh['P_max'], vtype=GRB.CONTINUOUS)
    P_batt = m.addVars(T, lb=-veh['P_batt_max'], ub=veh['P_batt_max'], vtype=GRB.CONTINUOUS)
    z = m.addVars(T, lb=0, ub=veh['S_MODES'], vtype=GRB.INTEGER)
    w = m.addVars(T, lb=0, vtype=GRB.CONTINUOUS) 
    
    m.addConstr(E[0] == veh['E_init'])
    for t in range(T):
        m.addConstr(E[t+1] <= E[t] - veh['tau'] * P_batt[t] - veh['gamma'] * P_batt[t]*P_batt[t])
        m.addConstr(P_eng[t] <= z[t] * (veh['P_max'] / veh['S_MODES']))
        m.addConstr(P_batt[t] + P_eng[t] >= D_profile[t])
        m.addConstr(w[t] >= z[t] - z_frac[t])
        m.addConstr(w[t] >= z_frac[t] - z[t])

    m.setObjective(gp.quicksum(w[t] for t in range(T)), GRB.MINIMIZE)
    m.optimize()
    
    if m.status == GRB.OPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
        return np.array([np.round(z[t].X) for t in range(T)])
    return np.clip(np.round(z_frac), 0, veh['S_MODES'])

def build_diffopt_layer(v_constraints, veh):
    E = cp.Variable(T+1)
    P_eng = cp.Variable(T, nonneg=True)
    P_batt = cp.Variable(T)
    z = cp.Variable(T) 
    s = cp.Variable(v_constraints, nonneg=True) 
    
    A_e_param = cp.Parameter((v_constraints, T+1))
    A_peng_param = cp.Parameter((v_constraints, T))
    A_pbatt_param = cp.Parameter((v_constraints, T))
    A_z_param = cp.Parameter((v_constraints, T))
    b_param = cp.Parameter(v_constraints)
    D_param = cp.Parameter(T, nonneg=True)
    
    constraints = [E[0] == veh['E_init'], z >= 0, z <= veh['S_MODES']]
    for t in range(T):
        constraints.append(E[t+1] - E[t] + veh['tau'] * P_batt[t] + veh['gamma'] * cp.square(P_batt[t]) <= 0)
        constraints.append(P_eng[t] <= z[t] * (veh['P_max'] / veh['S_MODES']))
        constraints.append(P_batt[t] + P_eng[t] >= D_param[t])
        constraints.append(E[t] <= veh['E_max'])
        constraints.append(E[t] >= 0)
        constraints.append(P_batt[t] >= -veh['P_batt_max'])
        constraints.append(P_batt[t] <= veh['P_batt_max'])
            
    constraints.append(E[T] <= veh['E_max'])
    constraints.append(E[T] >= 0)
    constraints.append(A_e_param @ E + A_peng_param @ P_eng + A_pbatt_param @ P_batt + A_z_param @ z <= b_param + s)
    
    obj = cp.Minimize(cp.sum(cp.multiply(veh['alpha'], cp.square(P_eng))) + 
                      cp.sum(cp.multiply(veh['beta'], z)) + 
                      veh['eta'] * (veh['E_max'] - E[T]) + 
                      10000.0 * cp.sum(s))
    
    prob = cp.Problem(obj, constraints)
    return CvxpyLayer(prob, parameters=[A_e_param, A_peng_param, A_pbatt_param, A_z_param, b_param, D_param], variables=[E, P_eng, P_batt, z, s])



# %%
# ==============================================================================
# 2. EXPERIMENT SWEEP EXECUTION - SUPERVISED
# ==============================================================================
N_TRAIN_LIST = [10, 25, 50, 80, 100]
V_CONSTRAINTS_LIST = [3, 5, 8]
HIDDEN_LAYERS_LIST = [(64, 64)]

os.makedirs("data", exist_ok=True)

results = {
    'std_nn': defaultdict(lambda: defaultdict(lambda: {'err_z': [], 'err_cont': [], 'opt_gap': [], 'solve_time': []})),
    'df_surrogate': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'err_z': [], 'err_cont': [], 'opt_gap': [], 'solve_time': []}))),
    'time_true': []
}

print(f"Beginning Cross-Instance Sweep (SUPERVISED)!\n")

for inst in range(N_INSTANCES):
    print(f"==================================================")
    print(f"  GENERATING VEHICLE INSTANCE {inst+1}/{N_INSTANCES}")
    print(f"==================================================")
    
    veh = {
        'eta': 2.0,
        'alpha': np.random.uniform(1.0, 3.0, T),
        'beta': np.random.uniform(0.5, 1.5, T),
        'E_init': np.random.uniform(5.0,8.0),
        'tau': 1.0,
        'P_max': 2.0,
        'E_max': 10.0,
        'P_batt_max': 3.0, 
        'gamma': 0.1,      
        'S_MODES': 3
    }
    
    veh['alpha_t'] = torch.tensor(veh['alpha'], dtype=torch.float32)
    veh['beta_t'] = torch.tensor(veh['beta'], dtype=torch.float32)

    # 2.2 Data Generation (Full tracking for supervised)
    data_file = f"data/supervised_inst_{inst}.npz"
    if os.path.exists(data_file):
        print(f"Loading cached labels from {data_file}")
        d = np.load(data_file)
        D_train, Z_train, E_train, Peng_train, Pbatt_train, Obj_train = d['D_train'], d['Z_train'], d['E_train'], d['Peng_train'], d['Pbatt_train'], d['Obj_train']
        D_test, Z_test, E_test, Peng_test, Pbatt_test, Obj_test = d['D_test'], d['Z_test'], d['E_test'], d['Peng_test'], d['Pbatt_test'], d['Obj_test']
    else:
        print(f"Generating Datasets for Instance {inst+1} (Train={N_TRAIN_MAX}, Test={N_TEST})...")
        D_train = np.random.uniform(0.5, 2.5, (N_TRAIN_MAX, T))
        D_test = np.random.uniform(0.5, 2.5, (N_TEST, T))
        
        Z_train = np.zeros((N_TRAIN_MAX, T))
        E_train = np.zeros((N_TRAIN_MAX, T+1))
        Peng_train = np.zeros((N_TRAIN_MAX, T))
        Pbatt_train = np.zeros((N_TRAIN_MAX, T))
        Obj_train = np.zeros(N_TRAIN_MAX)
        
        Z_test = np.zeros((N_TEST, T))
        E_test = np.zeros((N_TEST, T+1))
        Peng_test = np.zeros((N_TEST, T))
        Pbatt_test = np.zeros((N_TEST, T))
        Obj_test = np.zeros(N_TEST)

        for i in tqdm(range(N_TRAIN_MAX), desc="Train Gen"):
            Z_train[i], E_train[i], Peng_train[i], Pbatt_train[i], Obj_train[i] = solve_true_miqcqp_full(D_train[i], veh)
        for i in tqdm(range(N_TEST), desc="Test Gen"):
            Z_test[i], E_test[i], Peng_test[i], Pbatt_test[i], Obj_test[i] = solve_true_miqcqp_full(D_test[i], veh)
            
        np.savez(data_file, 
                 D_train=D_train, Z_train=Z_train, E_train=E_train, Peng_train=Peng_train, Pbatt_train=Pbatt_train, Obj_train=Obj_train,
                 D_test=D_test, Z_test=Z_test, E_test=E_test, Peng_test=Peng_test, Pbatt_test=Pbatt_test, Obj_test=Obj_test)
        print(f"Saved to {data_file}")
        
    true_continuous_sums = []
    Peng_true_list = []
    Pbatt_true_list = []
    for i in range(N_TEST):
        _, _, Peng_true, Pbatt_true = evaluate_fixed_z_miqcqp(D_test[i], veh, Z_test[i])
        Peng_true_list.append(Peng_true)
        Pbatt_true_list.append(Pbatt_true)
        true_continuous_sums.append(max(np.sum(np.abs(Peng_true)) + np.sum(np.abs(Pbatt_true)), 1e-5))

    D_test_t = torch.tensor(D_test, dtype=torch.float32)

    def evaluate_model_on_test(model, is_std_nn, v_constraints=None, cvx_layer=None):
        err_z, err_cont, opt_gap, solve_time = [], [], [], []
        for i in range(N_TEST):
            d_req = D_test[i]
            z_true = Z_test[i]
            true_obj = Obj_test[i]
            tcs = true_continuous_sums[i]
            
            with torch.no_grad():
                d_t_batch = D_test_t[i].unsqueeze(0)
                if is_std_nn:
                    z_frac = model(d_t_batch).squeeze(0).numpy()
                else:
                    theta = model(d_t_batch)
                    A_e_pred_t = theta[0, :v_constraints * (T+1)].view(v_constraints, T+1)
                    A_peng_pred_t = theta[0, v_constraints * (T+1):v_constraints * (2*T+1)].view(v_constraints, T)
                    A_pbatt_pred_t = theta[0, v_constraints * (2*T+1):v_constraints * (3*T+1)].view(v_constraints, T)
                    A_z_pred_t = theta[0, v_constraints * (3*T+1):v_constraints * (4*T+1)].view(v_constraints, T)
                    b_pred_t = theta[0, v_constraints * (4*T+1):]
            
            t0 = time.perf_counter()
            if not is_std_nn:
                try:
                    _, _, _, z_frac_t, _ = cvx_layer(A_e_pred_t, A_peng_pred_t, A_pbatt_pred_t, A_z_pred_t, b_pred_t, D_test_t[i])
                    z_frac = z_frac_t.detach().numpy()
                except Exception:
                    z_frac = np.zeros(T)
                    
            z_final = qi_zhang_miqcqp_restoration(d_req, veh, z_frac)
            solve_time.append(time.perf_counter() - t0)
            
            pred_obj, _, Peng, Pbatt = evaluate_fixed_z_miqcqp(d_req, veh, z_final)
            err_z.append(np.sum(np.abs(z_final - z_true)) / max(np.sum(z_true), 1e-5) * 100)
            opt_gap.append(abs(pred_obj - true_obj) / abs(true_obj) * 100 if pred_obj != float('inf') else 100.0)
            cont_error = (np.sum(np.abs(Peng - Peng_true_list[i])) + np.sum(np.abs(Pbatt - Pbatt_true_list[i]))) / tcs * 100
            err_cont.append(cont_error)
            
        return np.mean(err_z), np.mean(err_cont), np.mean(opt_gap), solve_time

    # 2.3 Train and Benchmark
    # Track times for true solutions
    true_time0 = time.perf_counter()
    for i in range(N_TEST):
        tt0 = time.perf_counter()
        solve_true_miqcqp_full(D_test[i], veh)
        results['time_true'].append(time.perf_counter() - tt0)
        
    for hl in HIDDEN_LAYERS_LIST:
        # SUPERVISED STANDARD NN
        for n in N_TRAIN_LIST:
            std_model = nn.Sequential(
                nn.Linear(T, hl[0]), nn.ReLU(),
                nn.Linear(hl[0], hl[1]), nn.ReLU(),
                nn.Linear(hl[1], T)
            )
            optimizer = optim.Adam(std_model.parameters(), lr=0.01)
            loss_fn = nn.MSELoss()
            
            D_tr = torch.tensor(D_train[:n], dtype=torch.float32)
            Z_tr = torch.tensor(Z_train[:n], dtype=torch.float32)
            
            for epoch in range(TRAINING_EPOCHS):
                optimizer.zero_grad()
                loss = loss_fn(std_model(D_tr), Z_tr)
                loss.backward()
                optimizer.step()
                
            ez, ec, og, st_list = evaluate_model_on_test(std_model, is_std_nn=True)
            results['std_nn'][hl][n]['err_z'].append(ez)
            results['std_nn'][hl][n]['err_cont'].append(ec)
            results['std_nn'][hl][n]['opt_gap'].append(og)
            results['std_nn'][hl][n]['solve_time'].extend(st_list)
            
            print(f"[{hl}] Std NN (N={n}) ErrZ={ez:.1f}%")
        
        # SUPERVISED DIFFOPT SURROGATE
        for v in V_CONSTRAINTS_LIST:
            cvx_layer = build_diffopt_layer(v, veh)
            for n in N_TRAIN_LIST:
                df_model = nn.Sequential(
                    nn.Linear(T, hl[0]), nn.Softplus(),
                    nn.Linear(hl[0], hl[1]), nn.Softplus(),
                    nn.Linear(hl[1], v * (4 * T + 1) + v)
                )
                optimizer = optim.Adam(df_model.parameters(), lr=0.01)
                mse_loss = nn.MSELoss()
                
                D_tr = torch.tensor(D_train[:n], dtype=torch.float32)
                Z_tr = torch.tensor(Z_train[:n], dtype=torch.float32)
                E_tr = torch.tensor(E_train[:n], dtype=torch.float32)
                Peng_tr = torch.tensor(Peng_train[:n], dtype=torch.float32)
                Pbatt_tr = torch.tensor(Pbatt_train[:n], dtype=torch.float32)
                
                for epoch in range(TRAINING_EPOCHS):
                    for i in range(n):
                        d_req = D_train[i]
                        d_t_batch = D_tr[i].unsqueeze(0)
                        theta = df_model(d_t_batch)
                        
                        A_e_pred = theta[0, :v * (T+1)].view(v, T+1)
                        A_peng_pred = theta[0, v * (T+1):v * (2*T+1)].view(v, T)
                        A_pbatt_pred = theta[0, v * (2*T+1):v * (3*T+1)].view(v, T)
                        A_z_pred = theta[0, v * (3*T+1):v * (4*T+1)].view(v, T)
                        b_pred = theta[0, v * (4*T+1):]
                        try:
                            E_frac, P_eng_frac, P_batt_frac, z_frac, s_frac = cvx_layer(
                                A_e_pred, A_peng_pred, A_pbatt_pred, A_z_pred, b_pred, D_tr[i], 
                                solver_args={'eps': 1e-3, 'max_iters': 2000})
                                
                            loss_z = 10.0 * mse_loss(z_frac, Z_tr[i])
                            loss_e = 1.0 * mse_loss(E_frac, E_tr[i])
                            loss_pe = 1.0 * mse_loss(P_eng_frac, Peng_tr[i])
                            loss_pb = 1.0 * mse_loss(P_batt_frac, Pbatt_tr[i])
                            
                            integrality_pen = torch.sum(torch.sin(np.pi * z_frac)**2)
                            loss_slack = 10.0 * torch.sum(s_frac)
                            
                            loss = loss_z + loss_e + loss_pe + loss_pb + 2.0 * integrality_pen + loss_slack
                            
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        except Exception:
                            pass
            
                ez, ec, og, st_list = evaluate_model_on_test(df_model, is_std_nn=False, v_constraints=v, cvx_layer=cvx_layer)
                results['df_surrogate'][hl][v][n]['err_z'].append(ez)
                results['df_surrogate'][hl][v][n]['err_cont'].append(ec)
                results['df_surrogate'][hl][v][n]['opt_gap'].append(og)
                results['df_surrogate'][hl][v][n]['solve_time'].extend(st_list)
                
                print(f"[{hl}] DF Surr SUPERVISED (V={v}, N={n}) ErrZ={ez:.1f}%")

with open('sweep_results_supervised.json', 'w') as f:
    json.dump(convert_to_json_serializable(results), f, indent=4)
print(f"\nSweep Finished! Cross-Instance Averages Computed & Saved.")

# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

with open('sweep_results_supervised.json', 'r') as f:
    res = json.load(f)

N_TRAIN_LIST = [10, 25, 50, 80, 100]
V_CONSTRAINTS_LIST = [3, 5, 8]
HIDDEN_LAYERS_LIST = ["(64, 64)"]

def create_boxplots(hl_choice, results_dict):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Cross-Instance Performance Distribution (Hidden Layers: {hl_choice}) - SUPERVISED", fontsize=16)
    
    metrics = ['err_z', 'err_cont', 'opt_gap']
    titles = ["Discrete Decision Error (%)", "Continuous Decision Error (%)", "Optimality Gap (%)"]
    ylabels = ["% Error", "% Error (MAE)", "% Gap"]

    positions = np.arange(len(N_TRAIN_LIST)) * (len(V_CONSTRAINTS_LIST) + 2)
    width = 0.8

    cmap = plt.colormaps['viridis'] if hasattr(plt, 'colormaps') else plt.cm.get_cmap('viridis')
    colors = cmap(np.linspace(0.1, 0.9, len(V_CONSTRAINTS_LIST)))
    
    for i in range(3):
        ax = axes[i]
        m = metrics[i]
        
        std_data = [results_dict['std_nn'][hl_choice][str(n)][m] for n in N_TRAIN_LIST]
        b_std = ax.boxplot(std_data, positions=positions, widths=width, patch_artist=True,
                           boxprops=dict(facecolor='lightgray'), medianprops=dict(color='black'))
        
        for idx, v in enumerate(V_CONSTRAINTS_LIST):
            df_data = [results_dict['df_surrogate'][hl_choice][str(v)][str(n)][m] for n in N_TRAIN_LIST]
            b_df = ax.boxplot(df_data, positions=positions + (idx + 1) * width, widths=width, patch_artist=True,
                              boxprops=dict(facecolor=colors[idx]), medianprops=dict(color='black'))
            
        ax.set_title(titles[i])
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel(ylabels[i])
        ax.set_xticks(positions + (len(V_CONSTRAINTS_LIST) * width) / 2)
        ax.set_xticklabels(N_TRAIN_LIST)
        ax.grid(True, linestyle=':', alpha=0.6, axis='y')
        
        if i == 0:
            patches = [mpatches.Patch(color='lightgray', label='Standard NN')]
            for idx, v in enumerate(V_CONSTRAINTS_LIST):
                patches.append(mpatches.Patch(color=colors[idx], label=f'DF Surrogate (V={v})'))
            ax.legend(handles=patches)
            
    ax_time = axes[3]
    y_vals = np.arange(1, len(results_dict.get('time_true', [])) + 1) / len(results_dict.get('time_true', [])) * 100 if len(results_dict.get('time_true', [])) > 0 else []
    
    true_time_cdf = np.sort(results_dict.get('time_true', []))
    ax_time.step(true_time_cdf, y_vals, label='True MIQCQP', where='post', lw=2, color='darkred')

    std_time_N100 = np.sort(results_dict['std_nn'][hl_choice]['100']['solve_time'])
    y_vals_std = np.arange(1, len(std_time_N100) + 1) / len(std_time_N100) * 100
    ax_time.step(std_time_N100, y_vals_std, label='Standard NN', where='post', lw=2, color='black', linestyle='--')
    
    for idx, v in enumerate(V_CONSTRAINTS_LIST):
        df_time_N100 = np.sort(results_dict['df_surrogate'][hl_choice][str(v)]['100']['solve_time'])
        y_vals_df = np.arange(1, len(df_time_N100) + 1) / len(df_time_N100) * 100
        ax_time.step(df_time_N100, y_vals_df, label=f'DF Surrogate (V={v})', where='post', lw=2, color=colors[idx])
        
    ax_time.set_title("Computational Eval Time CDF")
    ax_time.set_xlabel("Solution Time (seconds, log scale)")
    ax_time.set_ylabel("% of Instances Solved")
    ax_time.set_xscale('log')
    ax_time.legend()
    ax_time.grid(True, linestyle=':', alpha=0.6)
            
    plt.tight_layout()
    plt.show()

for hl in HIDDEN_LAYERS_LIST:
    create_boxplots(hl, res)

# %%

# %%
# !pip install numpy torch cvxpy cvxpylayers gurobipy matplotlib tqdm


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


warnings.filterwarnings("ignore", message="Solved/Inaccurate")

# ==============================================================================
# 0. HYPERPARAMETERS & PHYSICS SETUP
# ==============================================================================
T = 30                  
N_TRAIN_MAX = 100            
N_TEST = 100             
TRAINING_EPOCHS = 20
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

def solve_true_miqcqp(D_profile, veh):
    m, z, _, _, _ = build_gurobi_miqcqp(D_profile, veh)
    m.optimize()
    if m.status == GRB.OPTIMAL or m.status == GRB.TIME_LIMIT:
        if m.SolCount > 0:
            return np.array([z[t].X for t in range(T)]), m.ObjVal
    return np.zeros(T), float('inf')

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
# 2. EXPERIMENT SWEEP EXECUTION
# ==============================================================================
N_TRAIN_LIST = [10, 25, 50, 80, 100]
V_CONSTRAINTS_LIST = [3, 5, 8]
HIDDEN_LAYERS_LIST = [(16, 16)]

# We will accumulate metrics across N_INSTANCES (10 different vehicles).
results = {
    'std_nn': defaultdict(lambda: defaultdict(lambda: {'err_z': [], 'err_cont': [], 'opt_gap': [], 'solve_time': []})),
    'df_surrogate': defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'err_z': [], 'err_cont': [], 'opt_gap': [], 'solve_time': []}))),
    'time_true': []
}

print(f"Beginning Cross-Instance Sweep!\n")
with open('sweep_results.json', 'w') as f:
    json.dump(convert_to_json_serializable(results), f, indent=4)

for inst in range(N_INSTANCES):
    print(f"==================================================")
    print(f"  GENERATING VEHICLE INSTANCE {inst+1}/{N_INSTANCES}")
    print(f"==================================================")
    import os
    import pickle
    
    cache_file = f"dataset_instance_{inst}.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached dataset for Instance {inst+1} from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        veh = cache['veh']
        D_train = cache['D_train']
        D_test = cache['D_test']
        Z_train = cache['Z_train']
        Obj_train = cache['Obj_train']
        Z_test = cache['Z_test']
        Obj_test = cache['Obj_test']
        true_continuous_sums = cache['true_continuous_sums']
        Peng_true_list = cache['Peng_true_list']
        Pbatt_true_list = cache['Pbatt_true_list']
        time_true_list = cache['time_true_list']
        
    else:
        # Generate Paper Specifications
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
    
        # 2.2 Data Generation for this instance
        print(f"Generating Datasets for Instance {inst+1} (Train={N_TRAIN_MAX}, Test={N_TEST})...")
        D_train = np.random.uniform(0.5, 2.5, (N_TRAIN_MAX, T))
        D_test = np.random.uniform(0.5, 2.5, (N_TEST, T))
        Z_train = np.zeros((N_TRAIN_MAX, T))
        Obj_train = np.zeros(N_TRAIN_MAX)
        Z_test = np.zeros((N_TEST, T))
        Obj_test = np.zeros(N_TEST)
    
        for i in range(N_TRAIN_MAX):
            Z_train[i], Obj_train[i] = solve_true_miqcqp(D_train[i], veh)
            
        time_true_list = []
        for i in range(N_TEST):
            tt0 = time.perf_counter()
            Z_test[i], Obj_test[i] = solve_true_miqcqp(D_test[i], veh)
            time_true_list.append(time.perf_counter() - tt0)
            
        true_continuous_sums = []
        Peng_true_list = []
        Pbatt_true_list = []
        for i in range(N_TEST):
            _, _, Peng_true, Pbatt_true = evaluate_fixed_z_miqcqp(D_test[i], veh, Z_test[i])
            Peng_true_list.append(Peng_true)
            Pbatt_true_list.append(Pbatt_true)
            true_continuous_sums.append(max(np.sum(np.abs(Peng_true)) + np.sum(np.abs(Pbatt_true)), 1e-5))
            
        # Save to cache
        print(f"Saving generated dataset to cache: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'veh': veh, 'D_train': D_train, 'D_test': D_test, 
                'Z_train': Z_train, 'Obj_train': Obj_train, 
                'Z_test': Z_test, 'Obj_test': Obj_test, 
                'true_continuous_sums': true_continuous_sums, 
                'Peng_true_list': Peng_true_list, 'Pbatt_true_list': Pbatt_true_list,
                'time_true_list': time_true_list
            }, f)

    D_test_t = torch.tensor(D_test, dtype=torch.float32)

    def evaluate_model_on_test(model, is_std_nn, v_constraints=None, cvx_layer=None):
        err_z, err_cont, opt_gap, solve_time = [], [], [], []
        z_pred_list = []
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
            z_pred_list.append(z_final)
            
            pred_obj, _, Peng, Pbatt = evaluate_fixed_z_miqcqp(d_req, veh, z_final)
            err_z.append(np.sum(np.abs(z_final - z_true)) / max(np.sum(z_true), 1e-5) * 100)
            opt_gap.append(abs(pred_obj - true_obj) / abs(true_obj) * 100 if pred_obj != float('inf') else 100.0)
            cont_error = (np.sum(np.abs(Peng - Peng_true_list[i])) + np.sum(np.abs(Pbatt - Pbatt_true_list[i]))) / tcs * 100
            err_cont.append(cont_error)
            
        # Return full per-instance lists (NOT means) so box plots show real distributions
        return err_z, err_cont, opt_gap, solve_time, z_pred_list

    # 2.3 Train and Benchmark
    results['time_true'].extend(time_true_list)
        
    for hl in HIDDEN_LAYERS_LIST:
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
            
            dataset = torch.utils.data.TensorDataset(D_tr, Z_tr)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(16, n), shuffle=True)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
            
            for epoch in range(TRAINING_EPOCHS):
                for batch_D, batch_Z in dataloader:
                    optimizer.zero_grad()
                    loss = loss_fn(std_model(batch_D), batch_Z)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
            ez_list, ec_list, og_list, st_list, _ = evaluate_model_on_test(std_model, is_std_nn=True)
            results['std_nn'][hl][n]['err_z'].extend(ez_list)   # extend: all 100 test values go in
            results['std_nn'][hl][n]['err_cont'].extend(ec_list)
            results['std_nn'][hl][n]['opt_gap'].extend(og_list)
            results['std_nn'][hl][n]['solve_time'].extend(st_list)
            with open('sweep_results.json', 'w') as f:
                json.dump(convert_to_json_serializable(results), f, indent=4)
            
            print(f"[{hl}] Std NN (N={n}) ErrZ={np.mean(ez_list):.1f}%")
        
        for v in V_CONSTRAINTS_LIST:
            cvx_layer = build_diffopt_layer(v, veh)
            for n in N_TRAIN_LIST:
                df_model = nn.Sequential(
                    nn.Linear(T, hl[0]), nn.Softplus(),
                    nn.Linear(hl[0], hl[1]), nn.Softplus(),
                    nn.Linear(hl[1], v * (4 * T + 1) + v)
                )
                optimizer = optim.Adam(df_model.parameters(), lr=0.01)
                D_tr = torch.tensor(D_train[:n], dtype=torch.float32)
                
                dataset = torch.utils.data.TensorDataset(D_tr)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(16, n), shuffle=True)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

                for epoch in range(TRAINING_EPOCHS):
                    for batch in dataloader:
                        batch_D = batch[0]
                        batch_size = batch_D.size(0)
                        optimizer.zero_grad()
                        batch_loss = 0.0
                        valid_items = 0
                        
                        for i in range(batch_size):
                            d_t_batch = batch_D[i].unsqueeze(0)
                            theta = df_model(d_t_batch)
                            A_e_pred = theta[0, :v * (T+1)].view(v, T+1)
                            A_peng_pred = theta[0, v * (T+1):v * (2*T+1)].view(v, T)
                            A_pbatt_pred = theta[0, v * (2*T+1):v * (3*T+1)].view(v, T)
                            A_z_pred = theta[0, v * (3*T+1):v * (4*T+1)].view(v, T)
                            b_pred = theta[0, v * (4*T+1):]
                            try:
                                E_frac, P_eng_frac, P_batt_frac, z_frac, s_frac = cvx_layer(
                                    A_e_pred, A_peng_pred, A_pbatt_pred, A_z_pred, b_pred, batch_D[i], solver_args={'eps': 1e-3, 'max_iters': 2000})
                                obj_val = torch.sum(veh['alpha_t'] * P_eng_frac**2 + veh['beta_t'] * z_frac) + veh['eta'] * (veh['E_max'] - E_frac[-1])
                                integrality_pen = torch.sum(torch.sin(np.pi * z_frac)**2)
                                loss = obj_val + 20.0 * integrality_pen + 10000.0 * torch.sum(s_frac)
                                
                                batch_loss = batch_loss + loss
                                valid_items += 1
                            except Exception:
                                pass
                        
                        if valid_items > 0:
                            (batch_loss / valid_items).backward()
                            optimizer.step()
                            
                    scheduler.step()
            
                ez_list, ec_list, og_list, st_list, z_pred_list = evaluate_model_on_test(df_model, is_std_nn=False, v_constraints=v, cvx_layer=cvx_layer)
                results['df_surrogate'][hl][v][n]['err_z'].extend(ez_list)   # extend: all 100 test values go in
                results['df_surrogate'][hl][v][n]['err_cont'].extend(ec_list)
                results['df_surrogate'][hl][v][n]['opt_gap'].extend(og_list)
                results['df_surrogate'][hl][v][n]['solve_time'].extend(st_list)
                # Save z predictions for the best-trained model (N=100) for visualization
                if n == N_TRAIN_LIST[-1]:
                    results['df_surrogate'][hl][v]['z_pred_sample'] = [zp.tolist() for zp in z_pred_list[:6]]
                    results['df_surrogate'][hl][v]['z_true_sample'] = [Z_test[i].tolist() for i in range(6)]
                with open('sweep_results.json', 'w') as f:
                    json.dump(convert_to_json_serializable(results), f, indent=4)
                
                print(f"[{hl}] DF Surr (V={v}, N={n}) ErrZ={np.mean(ez_list):.1f}%")

print(f"\nSweep Finished! Cross-Instance Averages Computed.")
# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

with open('sweep_results.json', 'r') as f:
    res = json.load(f)

N_TRAIN_LIST = [10, 25, 50, 80, 100]
V_CONSTRAINTS_LIST = [3, 5, 8]
HIDDEN_LAYERS_LIST = ["(16, 16)"]

def create_boxplots(hl_choice, results_dict):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Cross-Instance Performance Distribution (Hidden Layers: {hl_choice})", fontsize=16)
    
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
        
        if i == 2:
            ax.set_ylim(-0.5, 17)
        
        if i == 0:
            patches = [mpatches.Patch(color='lightgray', label='Standard NN')]
            for idx, v in enumerate(V_CONSTRAINTS_LIST):
                patches.append(mpatches.Patch(color=colors[idx], label=f'DF Surrogate (V={v})'))
            ax.legend(handles=patches)
            
    # 4th Plot: Computational Eval Time CDF
    ax_time = axes[3]
    y_vals = np.arange(1, len(results_dict.get('time_true', [])) + 1) / len(results_dict.get('time_true', [])) * 100 if len(results_dict.get('time_true', [])) > 0 else []
    
    true_time_cdf = np.sort(results_dict.get('time_true', []))
    ax_time.step(true_time_cdf, y_vals, label='True MIQCQP', where='post', lw=2, color='darkred', alpha=0.9)

    std_time_N100 = np.sort(results_dict['std_nn'][hl_choice]['100']['solve_time'])
    y_vals_std = np.arange(1, len(std_time_N100) + 1) / len(std_time_N100) * 100
    ax_time.step(std_time_N100, y_vals_std, label='Standard NN', where='post', lw=2, color='black', linestyle='--', alpha=0.8)
    
    for idx, v in enumerate(V_CONSTRAINTS_LIST):
        df_time_N100 = np.sort(results_dict['df_surrogate'][hl_choice][str(v)]['100']['solve_time'])
        y_vals_df = np.arange(1, len(df_time_N100) + 1) / len(df_time_N100) * 100
        
        # Keep alpha 0.9 for V=3, else 0.5
        curve_alpha = 0.9 if v == 3 else 0.3
        
        ax_time.step(df_time_N100, y_vals_df, label=f'DF Surrogate (V={v})', where='post', lw=2, color=colors[idx], alpha=curve_alpha)
        
    ax_time.set_title("Computational Eval Time CDF")
    ax_time.set_xlabel("Solution Time (seconds, log scale)")
    ax_time.set_ylabel("% of Instances Solved")
    ax_time.set_xscale('log')
    ax_time.legend()
    ax_time.grid(True, linestyle=':', alpha=0.5)
            
    plt.tight_layout()
    # plt.savefig('results_plot_10_instances_1.png', dpi=600, bbox_inches='tight')
    plt.show()

for hl in HIDDEN_LAYERS_LIST:
    create_boxplots(hl, res)

# %%
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

with open('sweep_results.json', 'r') as f:
    res = json.load(f)

N_TRAIN_LIST = [10, 25, 50, 80, 100]
V_CONSTRAINTS_LIST = [3, 5, 8]
HIDDEN_LAYERS_LIST = ["(16, 16)"]

def create_boxplots(hl_choice, results_dict):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(f"Cross-Instance Performance Distribution (Hidden Layers: {hl_choice})", fontsize=16)
    
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
        
        # Standard NN Boxplots (Alpha = 0.5)
        std_data = [results_dict['std_nn'][hl_choice][str(n)][m] for n in N_TRAIN_LIST]
        b_std = ax.boxplot(std_data, positions=positions, widths=width, patch_artist=True,
                           boxprops=dict(facecolor='gray', alpha=1), medianprops=dict(color='black'))
        
        for idx, v in enumerate(V_CONSTRAINTS_LIST):
            df_data = [results_dict['df_surrogate'][hl_choice][str(v)][str(n)][m] for n in N_TRAIN_LIST]
            
            # Keep alpha 0.9 for V=3, else 0.5
            box_alpha = 0.9 if v == 3 else 0.4
            
            b_df = ax.boxplot(df_data, positions=positions + (idx + 1) * width, widths=width, patch_artist=True,
                              boxprops=dict(facecolor=colors[idx], alpha=box_alpha), medianprops=dict(color='black'))
            
        ax.set_title(titles[i])
        ax.set_xlabel("Number of Training Samples")
        ax.set_ylabel(ylabels[i])
        ax.set_xticks(positions + (len(V_CONSTRAINTS_LIST) * width) / 2)
        ax.set_xticklabels(N_TRAIN_LIST)
        ax.grid(True, linestyle=':', alpha=0.9, axis='y')
        
        if i == 2:
            ax.set_ylim(-0.5, 17)
        
        if i == 0:
            # Update legend to match the alpha values
            patches = [mpatches.Patch(color='gray', label='Standard NN', alpha=1)]
            for idx, v in enumerate(V_CONSTRAINTS_LIST):
                box_alpha = 0.9 if v == 3 else 0.4
                patches.append(mpatches.Patch(color=colors[idx], label=f'DF Surrogate (V={v})', alpha=box_alpha))
            ax.legend(handles=patches)
            
    # 4th Plot: Computational Eval Time CDF
    ax_time = axes[3]
    y_vals = np.arange(1, len(results_dict.get('time_true', [])) + 1) / len(results_dict.get('time_true', [])) * 100 if len(results_dict.get('time_true', [])) > 0 else []
    
    true_time_cdf = np.sort(results_dict.get('time_true', []))
    ax_time.step(true_time_cdf, y_vals, label='True MIQCQP', where='post', lw=2, color='darkred', alpha=0.9)

    std_time_N100 = np.sort(results_dict['std_nn'][hl_choice]['100']['solve_time'])
    y_vals_std = np.arange(1, len(std_time_N100) + 1) / len(std_time_N100) * 100
    ax_time.step(std_time_N100, y_vals_std, label='Standard NN', where='post', lw=2, color='black', linestyle='--', alpha=0.8)
    
    for idx, v in enumerate(V_CONSTRAINTS_LIST):
        df_time_N100 = np.sort(results_dict['df_surrogate'][hl_choice][str(v)]['100']['solve_time'])
        y_vals_df = np.arange(1, len(df_time_N100) + 1) / len(df_time_N100) * 100
        
        # Keep alpha 0.9 for V=3, else 0.5
        curve_alpha = 0.9 if v == 3 else 0.3
        
        ax_time.step(df_time_N100, y_vals_df, label=f'DF Surrogate (V={v})', where='post', lw=2, color=colors[idx], alpha=curve_alpha)
        
    ax_time.set_title("Computational Eval Time CDF")
    ax_time.set_xlabel("Solution Time (seconds, log scale)")
    ax_time.set_ylabel("% of Instances Solved")
    ax_time.set_xscale('log')
    ax_time.legend()
    ax_time.grid(True, linestyle=':', alpha=0.9)
            
    plt.tight_layout()
    # plt.savefig('results_plot_10_instances_new.png', dpi=600, bbox_inches='tight')
    plt.show()

for hl in HIDDEN_LAYERS_LIST:
    create_boxplots(hl, res)

# %%
# ==============================================================================
# Z Prediction vs Ground Truth Visualization
# ==============================================================================
def plot_z_predictions(hl_choice, results_dict, n_cases=6):
    V_LIST = [3, 5, 8]
    n_rows = len(V_LIST)
    fig, axes = plt.subplots(n_rows, n_cases, figsize=(n_cases * 3, n_rows * 2.8), sharey=True)
    fig.suptitle(f"Predicted vs True Integer Variable z (DF Surrogate, N=100, HL={hl_choice})", fontsize=14)
    
    time_steps = np.arange(T)
    
    for row, v in enumerate(V_LIST):
        v_key = str(v)
        z_preds = results_dict['df_surrogate'][hl_choice].get(v_key, {}).get('z_pred_sample', [])
        z_trues = results_dict['df_surrogate'][hl_choice].get(v_key, {}).get('z_true_sample', [])
        
        for col in range(n_cases):
            ax = axes[row][col]
            if col < len(z_preds):
                z_p = np.array(z_preds[col])
                z_t = np.array(z_trues[col])
                ax.step(time_steps, z_t, where='post', color='steelblue', lw=2, label='True z')
                ax.step(time_steps, z_p, where='post', color='tomato', lw=1.5,
                        linestyle='--', label='Pred z')
                err = np.sum(np.abs(z_p - z_t)) / max(np.sum(z_t), 1e-5) * 100
                ax.set_title(f"Case {col+1}\nErr={err:.1f}%", fontsize=8)
            ax.set_ylim(-0.3, 3.5)
            ax.set_yticks([0, 1, 2, 3])
            ax.grid(True, linestyle=':', alpha=0.5)
            if col == 0:
                ax.set_ylabel(f"V={v}\nz (mode)", fontsize=8)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc='upper left')
    
    for ax in axes[-1]:
        ax.set_xlabel("Time step t", fontsize=8)
    
    plt.tight_layout()
    # plt.savefig('z_predictions_plot_10_instances.png', dpi=600, bbox_inches='tight')
    plt.show()

for hl in HIDDEN_LAYERS_LIST:
    plot_z_predictions(hl, res)

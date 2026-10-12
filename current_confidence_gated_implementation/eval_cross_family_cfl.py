import os
import csv
import time
import argparse
import numpy as np
import torch
import ecole
import pyscipopt as scip

from model.model import GNNPolicy
from confidence_gated_brancher import UnifiedConfidenceGatedBranchingPolicy

def generate_cfl_instance(n_customers=50, n_facilities=25, seed=42, output_path=None):
    np.random.seed(seed)
    demands = np.random.randint(5, 36, size=n_customers)
    capacities = np.random.randint(20, 161, size=n_facilities)
    fixed_costs = np.random.randint(100, 200, size=n_facilities)
    
    # Coordinates in [0, 1]^2
    c_loc = np.random.rand(n_customers, 2)
    f_loc = np.random.rand(n_facilities, 2)
    trans_costs = np.linalg.norm(c_loc[:, None, :] - f_loc[None, :, :], axis=2) * 10.0

    model = scip.Model(f'CFL_{seed}')
    model.hideOutput(True)

    y = {}
    for i in range(n_facilities):
        y[i] = model.addVar(vtype='B', obj=float(fixed_costs[i]), name=f'y_{i}')
    
    x = {}
    for j in range(n_customers):
        for i in range(n_facilities):
            x[i, j] = model.addVar(vtype='C', lb=0.0, ub=1.0, obj=float(trans_costs[j, i]), name=f'x_{i}_{j}')
    
    # Customer demand satisfaction: sum_i x_ij == 1
    for j in range(n_customers):
        model.addCons(scip.quicksum(x[i, j] for i in range(n_facilities)) == 1.0)
    
    # Facility capacity constraints: sum_j d_j x_ij <= s_i y_i
    for i in range(n_facilities):
        model.addCons(scip.quicksum(demands[j] * x[i, j] for j in range(n_customers)) <= float(capacities[i]) * y[i])

    # Tightening cuts: x_ij <= y_i
    for j in range(n_customers):
        for i in range(n_facilities):
            model.addCons(x[i, j] <= y[i])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        model.writeProblem(output_path)
    
    return output_path

def run_cfl_benchmark(n_instances=5, n_customers=50, n_facilities=25, time_limit=30):
    os.makedirs('data/instances/cfl', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    instance_paths = []
    for i in range(n_instances):
        p = f'data/instances/cfl/instance_{i+1}.lp'
        if not os.path.exists(p):
            generate_cfl_instance(n_customers=n_customers, n_facilities=n_facilities, seed=100 + i, output_path=p)
        instance_paths.append(p)

    device = 'cpu'
    model_path = 'model/setcover/0/train_params.pkl'
    gnn = GNNPolicy().to(device)
    gnn.load_state_dict(torch.load(model_path, map_location=device))
    gnn.eval()

    csv_path = f"results/cross_family_cfl_eval_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fieldnames = [
        'instance_id', 'system', 'walltime', 'stime', 'nnodes', 'gap', 'status',
        'fallback_rate', 'is_ood', 'total_decisions', 'gnn_decisions', 'fallback_decisions'
    ]

    print("=" * 75)
    print("CROSS-FAMILY GENERALIZATION BENCHMARK: CAPACITATED FACILITY LOCATION (CFL)")
    print(f"Model trained on: SET COVERING | Evaluated on: CAPACITATED FACILITY LOCATION")
    print(f"Instances: {n_instances} ({n_facilities} facs x {n_customers} custs) | Time Limit: {time_limit}s")
    print(f"Results CSV: {csv_path}")
    print("=" * 75)

    results = []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, inst_path in enumerate(instance_paths, 1):
            print(f"\n--- [CFL Instance {idx}/{n_instances}] {inst_path} ---")

            # 1. SCIP Default
            m_scip = ecole.scip.Model.from_file(inst_path)
            m_scip.set_params({
                'limits/time': time_limit,
                'timing/clocktype': 1,
                'separating/maxrounds': 0,
                'presolving/maxrestarts': 0
            })
            pyscip = m_scip.as_pyscipopt()
            t0 = time.perf_counter()
            pyscip.optimize()
            scip_wall = time.perf_counter() - t0

            rec_scip = {
                'instance_id': idx,
                'system': 'SCIP_Default',
                'walltime': scip_wall,
                'stime': pyscip.getSolvingTime(),
                'nnodes': pyscip.getNNodes(),
                'gap': pyscip.getGap(),
                'status': pyscip.getStatus(),
                'fallback_rate': 0.0,
                'is_ood': False,
                'total_decisions': 0,
                'gnn_decisions': 0,
                'fallback_decisions': 0
            }
            writer.writerow(rec_scip)
            results.append(rec_scip)
            print(f"  [SCIP Default] : Nodes: {rec_scip['nnodes']:4d} | Time: {rec_scip['walltime']:5.2f}s | Gap: {rec_scip['gap']:.4f} | Status: {rec_scip['status']}")

            # 2. Pure GNN Zero-Shot
            env_pure = ecole.environment.Branching(
                observation_function={'node_obs': ecole.observation.NodeBipartite(), 'pseudocosts': ecole.observation.Pseudocosts()},
                scip_params={'limits/time': time_limit, 'timing/clocktype': 1, 'separating/maxrounds': 0, 'presolving/maxrestarts': 0}
            )
            policy_pure = UnifiedConfidenceGatedBranchingPolicy(gnn, mode='always_gnn')
            obs, act_set, _, done, _ = env_pure.reset(inst_path)
            t0 = time.perf_counter()
            d_cnt = 0
            while not done:
                action = policy_pure.select_action(obs['node_obs'], act_set, obs['pseudocosts'], depth=d_cnt)
                obs, act_set, _, done, _ = env_pure.step(action)
                d_cnt += 1
            pure_wall = time.perf_counter() - t0
            m_pure = env_pure.model.as_pyscipopt()

            rec_pure = {
                'instance_id': idx,
                'system': 'Pure_GNN_ZeroShot',
                'walltime': pure_wall,
                'stime': m_pure.getSolvingTime(),
                'nnodes': m_pure.getNNodes(),
                'gap': m_pure.getGap(),
                'status': m_pure.getStatus(),
                'fallback_rate': 0.0,
                'is_ood': False,
                'total_decisions': policy_pure.total_decisions,
                'gnn_decisions': policy_pure.gnn_decisions,
                'fallback_decisions': 0
            }
            writer.writerow(rec_pure)
            results.append(rec_pure)
            print(f"  [Pure GNN]     : Nodes: {rec_pure['nnodes']:4d} | Time: {rec_pure['walltime']:5.2f}s | Gap: {rec_pure['gap']:.4f} | Status: {rec_pure['status']}")

            # 3. Fixed Gated (No OOD Gate)
            env_fixed = ecole.environment.Branching(
                observation_function={'node_obs': ecole.observation.NodeBipartite(), 'pseudocosts': ecole.observation.Pseudocosts()},
                scip_params={'limits/time': time_limit, 'timing/clocktype': 1, 'separating/maxrounds': 0, 'presolving/maxrestarts': 0}
            )
            policy_fixed = UnifiedConfidenceGatedBranchingPolicy(
                gnn, mode='confidence', margin_thresh=0.02, entropy_thresh=0.85,
                enable_ood_detection=False, sparsify_ego_subgraph=False
            )
            obs, act_set, _, done, _ = env_fixed.reset(inst_path)
            t0 = time.perf_counter()
            d_cnt = 0
            while not done:
                action = policy_fixed.select_action(obs['node_obs'], act_set, obs['pseudocosts'], depth=d_cnt)
                obs, act_set, _, done, _ = env_fixed.step(action)
                d_cnt += 1
            fixed_wall = time.perf_counter() - t0
            m_fixed = env_fixed.model.as_pyscipopt()
            fb_fixed = (policy_fixed.fallback_decisions / max(1, policy_fixed.total_decisions)) * 100.0

            rec_fixed = {
                'instance_id': idx,
                'system': 'Fixed_Gated_NoOOD',
                'walltime': fixed_wall,
                'stime': m_fixed.getSolvingTime(),
                'nnodes': m_fixed.getNNodes(),
                'gap': m_fixed.getGap(),
                'status': m_fixed.getStatus(),
                'fallback_rate': fb_fixed,
                'is_ood': False,
                'total_decisions': policy_fixed.total_decisions,
                'gnn_decisions': policy_fixed.gnn_decisions,
                'fallback_decisions': policy_fixed.fallback_decisions
            }
            writer.writerow(rec_fixed)
            results.append(rec_fixed)
            print(f"  [Fixed Gate]   : Nodes: {rec_fixed['nnodes']:4d} | Time: {rec_fixed['walltime']:5.2f}s | Gap: {rec_fixed['gap']:.4f} | Status: {rec_fixed['status']} | Fallback: {fb_fixed:.1f}%")

            # 4. Full-Stack Flagship (Ours: Adaptive OOD + Ego-Subgraph + Triage + Calibrated)
            env_ours = ecole.environment.Branching(
                observation_function={'node_obs': ecole.observation.NodeBipartite(), 'pseudocosts': ecole.observation.Pseudocosts()},
                scip_params={'limits/time': time_limit, 'timing/clocktype': 1, 'separating/maxrounds': 0, 'presolving/maxrestarts': 0}
            )
            policy_ours = UnifiedConfidenceGatedBranchingPolicy(
                gnn, mode='confidence', margin_thresh=0.02, entropy_thresh=0.85,
                enable_ood_detection=True, sparsify_ego_subgraph=True,
                quantize_int8=True, enable_primal_diving=True
            )
            obs, act_set, _, done, _ = env_ours.reset(inst_path)
            t0 = time.perf_counter()
            d_cnt = 0
            while not done:
                action = policy_ours.select_action(obs['node_obs'], act_set, obs['pseudocosts'], depth=d_cnt)
                obs, act_set, _, done, _ = env_ours.step(action)
                d_cnt += 1
            ours_wall = time.perf_counter() - t0
            m_ours = env_ours.model.as_pyscipopt()
            fb_ours = (policy_ours.fallback_decisions / max(1, policy_ours.total_decisions)) * 100.0

            rec_ours = {
                'instance_id': idx,
                'system': 'Flagship_Ours',
                'walltime': ours_wall,
                'stime': m_ours.getSolvingTime(),
                'nnodes': m_ours.getNNodes(),
                'gap': m_ours.getGap(),
                'status': m_ours.getStatus(),
                'fallback_rate': fb_ours,
                'is_ood': policy_ours.is_ood_detected,
                'total_decisions': policy_ours.total_decisions,
                'gnn_decisions': policy_ours.gnn_decisions,
                'fallback_decisions': policy_ours.fallback_decisions
            }
            writer.writerow(rec_ours)
            results.append(rec_ours)
            print(f"  [Flagship Ours]: Nodes: {rec_ours['nnodes']:4d} | Time: {rec_ours['walltime']:5.2f}s | Gap: {rec_ours['gap']:.4f} | Status: {rec_ours['status']} | Fallback: {fb_ours:.1f}% (OOD={policy_ours.is_ood_detected})")

            f.flush()

    print("\n" + "=" * 75)
    print(f"CFL Benchmark completed successfully! Results written to: {csv_path}")
    print("=" * 75)
    return csv_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-instances', type=int, default=3)
    parser.add_argument('--n-customers', type=int, default=50)
    parser.add_argument('--n-facilities', type=int, default=25)
    parser.add_argument('--time-limit', type=int, default=30)
    args = parser.parse_args()

    run_cfl_benchmark(
        n_instances=args.n_instances,
        n_customers=args.n_customers,
        n_facilities=args.n_facilities,
        time_limit=args.time_limit
    )

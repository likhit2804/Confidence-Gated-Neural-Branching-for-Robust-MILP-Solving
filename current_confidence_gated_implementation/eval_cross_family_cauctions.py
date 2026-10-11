import os
import csv
import time
import ecole
import numpy as np
import torch

from model.model import GNNPolicy
from confidence_gated_brancher import UnifiedConfidenceGatedBranchingPolicy

def run_cross_family_benchmark(n_instances=5, time_limit=30):
    os.makedirs('data/instances/cauctions', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Generate reproducible instances if not present
    gen = ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500)
    gen.seed(42)
    instance_paths = []
    for i in range(n_instances):
        p = f'data/instances/cauctions/instance_{i+1}.lp'
        if not os.path.exists(p):
            m = next(gen)
            m.write_problem(p)
        instance_paths.append(p)

    device = 'cpu'
    gnn = GNNPolicy().to(device)
    gnn.load_state_dict(torch.load('model/setcover/0/train_params.pkl', map_location=device))
    gnn.eval()

    csv_path = f"results/cross_family_cauctions_eval_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fieldnames = [
        'instance_id', 'system', 'nnodes', 'stime', 'walltime', 'gap', 'status',
        'fallback_rate', 'total_decisions', 'gnn_decisions', 'fallback_decisions'
    ]

    print("=" * 70)
    print("CROSS-FAMILY ZERO-SHOT GENERALIZATION BENCHMARK: COMBINATORIAL AUCTION")
    print(f"Model trained on: SET COVERING | Evaluated on: COMBINATORIAL AUCTION")
    print(f"Total instances: {n_instances} | Time limit per solve: {time_limit}s")
    print(f"Results CSV: {csv_path}")
    print("=" * 70)

    results = []

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, inst_path in enumerate(instance_paths, 1):
            print(f"\n--- [Instance {idx}/{n_instances}] {inst_path} ---")

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
                'nnodes': pyscip.getNNodes(),
                'stime': pyscip.getSolvingTime(),
                'walltime': scip_wall,
                'gap': pyscip.getGap(),
                'status': pyscip.getStatus(),
                'fallback_rate': 0.0,
                'total_decisions': 0,
                'gnn_decisions': 0,
                'fallback_decisions': 0
            }
            writer.writerow(rec_scip)
            results.append(rec_scip)
            print(f"  [SCIP Default] : Nodes: {rec_scip['nnodes']:4d} | Time: {rec_scip['walltime']:5.2f}s | Gap: {rec_scip['gap']:.4f} | Status: {rec_scip['status']}")

            # 2. Pure GNN (Zero-Shot Cross Family)
            env_pure = ecole.environment.Branching(
                observation_function={'node_obs': ecole.observation.NodeBipartite(), 'pseudocosts': ecole.observation.Pseudocosts()},
                scip_params={'limits/time': time_limit, 'timing/clocktype': 1, 'separating/maxrounds': 0, 'presolving/maxrestarts': 0}
            )
            policy_pure = UnifiedConfidenceGatedBranchingPolicy(gnn, mode='always_gnn')
            obs, act_set, _, done, _ = env_pure.reset(inst_path)
            t0 = time.perf_counter()
            while not done:
                action = policy_pure.select_action(obs['node_obs'], act_set, obs['pseudocosts'])
                obs, act_set, _, done, _ = env_pure.step(action)
            pure_wall = time.perf_counter() - t0
            m_pure = env_pure.model.as_pyscipopt()

            rec_pure = {
                'instance_id': idx,
                'system': 'Pure_GNN_ZeroShot',
                'nnodes': m_pure.getNNodes(),
                'stime': m_pure.getSolvingTime(),
                'walltime': pure_wall,
                'gap': m_pure.getGap(),
                'status': m_pure.getStatus(),
                'fallback_rate': 0.0,
                'total_decisions': policy_pure.total_decisions,
                'gnn_decisions': policy_pure.gnn_decisions,
                'fallback_decisions': 0
            }
            writer.writerow(rec_pure)
            results.append(rec_pure)
            print(f"  [Pure GNN]     : Nodes: {rec_pure['nnodes']:4d} | Time: {rec_pure['walltime']:5.2f}s | Gap: {rec_pure['gap']:.4f} | Status: {rec_pure['status']}")

            # 3. Confidence-Gated (Ours, Universal Gate h=0.85)
            env_gated = ecole.environment.Branching(
                observation_function={'node_obs': ecole.observation.NodeBipartite(), 'pseudocosts': ecole.observation.Pseudocosts()},
                scip_params={'limits/time': time_limit, 'timing/clocktype': 1, 'separating/maxrounds': 0, 'presolving/maxrestarts': 0}
            )
            policy_gated = UnifiedConfidenceGatedBranchingPolicy(
                gnn, mode='confidence', margin_thresh=0.02, entropy_thresh=0.85,
                quantize_int8=True, enable_primal_diving=True
            )
            obs, act_set, _, done, _ = env_gated.reset(inst_path)
            t0 = time.perf_counter()
            while not done:
                action = policy_gated.select_action(obs['node_obs'], act_set, obs['pseudocosts'])
                obs, act_set, _, done, _ = env_gated.step(action)
            gated_wall = time.perf_counter() - t0
            m_gated = env_gated.model.as_pyscipopt()
            fb_rate = (policy_gated.fallback_decisions / max(1, policy_gated.total_decisions)) * 100.0

            rec_gated = {
                'instance_id': idx,
                'system': 'Confidence_Gated_Ours',
                'nnodes': m_gated.getNNodes(),
                'stime': m_gated.getSolvingTime(),
                'walltime': gated_wall,
                'gap': m_gated.getGap(),
                'status': m_gated.getStatus(),
                'fallback_rate': fb_rate,
                'total_decisions': policy_gated.total_decisions,
                'gnn_decisions': policy_gated.gnn_decisions,
                'fallback_decisions': policy_gated.fallback_decisions
            }
            writer.writerow(rec_gated)
            results.append(rec_gated)
            print(f"  [Gated Ours]   : Nodes: {rec_gated['nnodes']:4d} | Time: {rec_gated['walltime']:5.2f}s | Gap: {rec_gated['gap']:.4f} | Status: {rec_gated['status']} | Fallback: {fb_rate:.1f}%")

            f.flush()

    print("\n" + "=" * 70)
    print(f"Benchmark completed successfully! Results written to: {csv_path}")
    print("=" * 70)
    return csv_path

if __name__ == '__main__':
    run_cross_family_benchmark(n_instances=5, time_limit=30)

import os
import csv
import time
import argparse
import numpy as np
import torch
import ecole

from model.model import GNNPolicy
from confidence_gated_brancher import UnifiedConfidenceGatedBranchingPolicy

def shifted_geometric_mean(values, shift=1.0):
    val_arr = np.array(values, dtype=np.float64)
    return float(np.exp(np.mean(np.log(val_arr + shift))) - shift)

def calculate_par_k(times, statuses, time_limit, penalty=10):
    par_times = []
    for t, s in zip(times, statuses):
        if s.lower() == 'optimal':
            par_times.append(t)
        else:
            par_times.append(time_limit * penalty)
    return float(np.mean(par_times))

def run_full_stack_benchmark(n_instances=10, time_limit=180, tiers='small,medium,big', seed=0):
    device = 'cpu'
    model_path = 'model/setcover/0/train_params.pkl'
    print(f"Loading GNN model from {model_path}...")
    gnn = GNNPolicy().to(device)
    gnn.load_state_dict(torch.load(model_path, map_location=device))
    gnn.eval()

    # Apply Pillar 4: INT8 dynamic quantization
    print("[PILLAR 4] Applying PyTorch INT8 Dynamic Quantization...")
    gnn_quant = torch.quantization.quantize_dynamic(gnn, {torch.nn.Linear}, dtype=torch.qint8)

    # Initialize Full-Stack Flagship Policy (All 6 Pillars Active)
    policy = UnifiedConfidenceGatedBranchingPolicy(
        gnn_model=gnn_quant,
        margin_thresh=0.02,
        entropy_thresh=0.85,
        temperature=1.2,
        cardinality_scaling=True,       # Pillar 2
        enable_triage=True,             # Pillar 1
        early_exit_depth=20,
        pseudo_margin=0.35,
        frac_margin=0.20,
        quantize_int8=True,             # Pillar 4
        candidate_topk=50,              # Pillar 4
        enable_primal_diving=True,      # Pillar 3
        dive_entropy_thresh=0.35,
        dive_margin_thresh=0.25,
        enable_ood_detection=True,      # Phase A
        sparsify_ego_subgraph=True,     # Phase C
        mode='confidence',
        device=device
    )

    instances = []
    selected_tiers = [t.strip().lower() for t in tiers.split(',')]
    for t in selected_tiers:
        if t == 'small':
            instances += [{'tier': 'small', 'path': f'data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp'} for i in range(n_instances)]
        elif t == 'medium':
            instances += [{'tier': 'medium', 'path': f'data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp'} for i in range(n_instances)]
        elif t == 'big':
            instances += [{'tier': 'big', 'path': f'data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp'} for i in range(n_instances)]

    os.makedirs('results', exist_ok=True)
    csv_path = f"results/full_stack_flagship_eval_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fieldnames = [
        'instance_id', 'tier', 'instance_path', 'status', 'walltime', 'stime', 'nnodes', 'nlps',
        'primal_bound', 'dual_bound', 'gap', 'fallback_rate', 'early_exit_rate',
        'dive_triggers', 'is_ood', 'mean_norm_entropy', 'mean_margin'
    ]

    scip_parameters = {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        'limits/time': time_limit,
        'timing/clocktype': 1,
        'branching/vanillafullstrong/idempotent': True,
        'heuristics/rens/freq': 5,
        'heuristics/rins/freq': 5,
        'heuristics/guideddiving/freq': 5,
        'heuristics/linesearchdiving/freq': 5
    }

    obs_functions = {
        'node_obs': ecole.observation.NodeBipartite(),
        'pseudocosts': ecole.observation.Pseudocosts()
    }

    print("=" * 80)
    print("FULL-STACK FLAGSHIP BENCHMARK: ALL 6 PILLARS & ENHANCEMENTS ACTIVE")
    print(f"Pillars: Triage (P1) | Temp (P2) | Diving (P3) | INT8+TopK (P4) | Root OOD (Phase A) | Ego-Subgraph (Phase C)")
    print(f"Total Instances: {len(instances)} | Time Limit: {time_limit}s | Seed: {seed}")
    print(f"Results CSV: {csv_path}")
    print("=" * 80)

    records = []
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, inst in enumerate(instances, 1):
            tier = inst['tier']
            p = inst['path']
            print(f"\n[{idx}/{len(instances)}] [{tier.upper()}] {p}")

            policy.reset_stats()
            env = ecole.environment.Branching(
                observation_function=obs_functions,
                scip_params=scip_parameters
            )
            env.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = time.perf_counter()
            obs, act_set, _, done, _ = env.reset(p)
            depth_counter = 0

            while not done:
                node_obs = obs['node_obs']
                pseudocosts = obs['pseudocosts']
                action = policy.select_action(node_obs, act_set, pseudocosts, depth=depth_counter)
                obs, act_set, _, done, _ = env.step(action)
                depth_counter += 1

            walltime = time.perf_counter() - t0
            m = env.model.as_pyscipopt()

            fb_rate = (policy.fallback_decisions / max(1, policy.total_decisions)) * 100.0
            ee_rate = (policy.early_exit_decisions / max(1, policy.total_decisions)) * 100.0
            mean_norm_ent = float(np.mean(policy.norm_entropies)) if policy.norm_entropies else 0.0
            mean_marg = float(np.mean(policy.margins)) if policy.margins else 1.0

            rec = {
                'instance_id': idx,
                'tier': tier,
                'instance_path': p,
                'status': m.getStatus(),
                'walltime': walltime,
                'stime': m.getSolvingTime(),
                'nnodes': m.getNNodes(),
                'nlps': m.getNLPs(),
                'primal_bound': m.getPrimalbound(),
                'dual_bound': m.getDualbound(),
                'gap': m.getGap(),
                'fallback_rate': fb_rate,
                'early_exit_rate': ee_rate,
                'dive_triggers': policy.dive_triggers,
                'is_ood': policy.is_ood_detected,
                'mean_norm_entropy': mean_norm_ent,
                'mean_margin': mean_marg
            }
            writer.writerow(rec)
            f.flush()
            records.append(rec)

            print(f"   Nodes: {rec['nnodes']:4d} | Time: {walltime:5.2f}s | Gap: {rec['gap']:.4f} | Primal: {rec['primal_bound']:.2f} | Status: {rec['status']} | Fallback: {fb_rate:.1f}%")

    print("\n" + "=" * 80)
    print("FULL-STACK BENCHMARK SUMMARY SCORECARD")
    print("=" * 80)
    for tier in selected_tiers:
        tier_recs = [r for r in records if r['tier'] == tier]
        if not tier_recs:
            continue
        times = [r['walltime'] for r in tier_recs]
        nodes = [r['nnodes'] for r in tier_recs]
        statuses = [r['status'] for r in tier_recs]
        solved = sum(1 for s in statuses if s.lower() == 'optimal')
        geom_time = shifted_geometric_mean(times, shift=1.0)
        geom_nodes = shifted_geometric_mean(nodes, shift=10.0)
        par10 = calculate_par_k(times, statuses, time_limit, penalty=10)

        print(f"[{tier.upper()}] Solved: {solved}/{len(tier_recs)} ({solved/len(tier_recs)*100:.1f}%) | Mean Walltime: {np.mean(times):.2f}s | SGM Time (s=1): {geom_time:.2f}s | SGM Nodes (s=10): {geom_nodes:.1f} | PAR-10: {par10:.2f}s")

    return csv_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-instances', type=int, default=10)
    parser.add_argument('--time-limit', type=int, default=180)
    parser.add_argument('--tiers', default='small,medium,big')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    run_full_stack_benchmark(
        n_instances=args.n_instances,
        time_limit=args.time_limit,
        tiers=args.tiers,
        seed=args.seed
    )

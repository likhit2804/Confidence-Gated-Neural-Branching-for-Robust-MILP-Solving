import os
import sys
import argparse
import csv
import numpy as np
import time
import pathlib
import ecole
import pyscipopt
import torch
import torch.nn.functional as F

from model.model import GNNPolicy

def run_evaluation(problem='setcover', n_instances=20, seed=0, time_limit=300):
    result_file = f"eval_{problem}_{n_instances}inst_seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', result_file)

    device = 'cpu'
    print(f"=== Starting Rigorous Benchmark Evaluation ===")
    print(f"Problem: {problem}")
    print(f"Instances per size category: {n_instances}")
    print(f"Per-instance time limit: {time_limit}s")
    print(f"Device: {device}")

    # Build instance list (Small, Medium, Big)
    instances = []
    if problem == 'setcover':
        instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
    else:
        raise NotImplementedError(f"Problem {problem} not configured")

    # Load trained GNN model
    model_path = f"model/{problem}/{seed}/train_params.pkl"
    print(f"Loading GNN model from {model_path}...")
    gnn_model = GNNPolicy().to(device)
    gnn_model.load_state_dict(torch.load(model_path, map_location=device))
    gnn_model.eval()

    fieldnames = [
        'instance_type', 'instance_path', 'policy',
        'nnodes', 'nlps', 'stime', 'walltime', 'gap', 'status',
        'mean_entropy', 'mean_margin'
    ]

    scip_parameters = {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        'limits/time': time_limit,
        'timing/clocktype': 1,
        'branching/vanillafullstrong/idempotent': True
    }

    records = []

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        total = len(instances)
        for idx, inst in enumerate(instances, 1):
            inst_type = inst['type']
            inst_path = inst['path']
            print(f"\n[{idx}/{total}] [{inst_type.upper()}] {inst_path}")

            # 1. SCIP Default (Reliable Pseudocosts)
            scip_env = ecole.environment.Configuring(scip_params={
                **scip_parameters,
                'branching/relpscost/priority': 9999999
            })
            scip_env.seed(seed)
            t0 = time.perf_counter()
            scip_env.reset(inst_path)
            _, _, _, _, _ = scip_env.step({})
            scip_wall = time.perf_counter() - t0

            scip_m = scip_env.model.as_pyscipopt()
            scip_rec = {
                'instance_type': inst_type,
                'instance_path': inst_path,
                'policy': 'scip_default',
                'nnodes': scip_m.getNNodes(),
                'nlps': scip_m.getNLPs(),
                'stime': scip_m.getSolvingTime(),
                'walltime': scip_wall,
                'gap': scip_m.getGap(),
                'status': scip_m.getStatus(),
                'mean_entropy': -1.0,
                'mean_margin': -1.0
            }
            writer.writerow(scip_rec)
            csvfile.flush()
            records.append(scip_rec)
            print(f"   SCIP Default : {scip_rec['nnodes']:4d} nodes | {scip_rec['walltime']:.2f}s | status: {scip_rec['status']}")

            # 2. Learned GNN Policy
            gnn_env = ecole.environment.Branching(
                observation_function=ecole.observation.NodeBipartite(),
                scip_params=scip_parameters
            )
            gnn_env.seed(seed)
            torch.manual_seed(seed)

            node_entropies = []
            node_margins = []

            t0 = time.perf_counter()
            obs, act_set, _, done, _ = gnn_env.reset(inst_path)
            while not done:
                with torch.no_grad():
                    c_feat = torch.from_numpy(obs.row_features.astype(np.float32)).to(device)
                    e_idx = torch.from_numpy(obs.edge_features.indices.astype(np.int64)).to(device)
                    e_attr = torch.from_numpy(obs.edge_features.values.astype(np.float32)).view(-1, 1).to(device)
                    v_feat = torch.from_numpy(obs.variable_features.astype(np.float32)).to(device)

                    logits = gnn_model(c_feat, e_idx, e_attr, v_feat)
                    cand_logits = logits[act_set.astype(np.int64)]

                    # Calculate confidence signals
                    probs = F.softmax(cand_logits, dim=-1)
                    entropy = (-probs * torch.log(probs + 1e-12)).sum().item()
                    node_entropies.append(entropy)

                    if len(cand_logits) > 1:
                        top2 = torch.topk(probs, 2).values
                        margin = (top2[0] - top2[1]).item()
                    else:
                        margin = 1.0
                    node_margins.append(margin)

                    action = act_set[cand_logits.argmax().item()]
                    obs, act_set, _, done, _ = gnn_env.step(action)

            gnn_wall = time.perf_counter() - t0
            gnn_m = gnn_env.model.as_pyscipopt()
            mean_ent = float(np.mean(node_entropies)) if node_entropies else 0.0
            mean_marg = float(np.mean(node_margins)) if node_margins else 1.0

            gnn_rec = {
                'instance_type': inst_type,
                'instance_path': inst_path,
                'policy': 'gnn_supervised',
                'nnodes': gnn_m.getNNodes(),
                'nlps': gnn_m.getNLPs(),
                'stime': gnn_m.getSolvingTime(),
                'walltime': gnn_wall,
                'gap': gnn_m.getGap(),
                'status': gnn_m.getStatus(),
                'mean_entropy': mean_ent,
                'mean_margin': mean_marg
            }
            writer.writerow(gnn_rec)
            csvfile.flush()
            records.append(gnn_rec)
            print(f"   GNN Policy   : {gnn_rec['nnodes']:4d} nodes | {gnn_rec['walltime']:.2f}s | H: {mean_ent:.2f} | Marg: {mean_marg:.2f}")

    print(f"\n=== Full Evaluation Complete! Results saved to {csv_path} ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='setcover')
    parser.add_argument('--n-instances', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time-limit', type=int, default=300)
    args = parser.parse_args()

    run_evaluation(args.problem, args.n_instances, args.seed, args.time_limit)

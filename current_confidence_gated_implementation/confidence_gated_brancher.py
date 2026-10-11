import os
import sys
import argparse
import csv
import numpy as np
import time
import ecole
import pyscipopt
import torch
import torch.nn.functional as F

from model.model import GNNPolicy

class ConfidenceGatedBranchingPolicy:
    """
    Confidence-Gated Neural Branching Policy for MILP Solving.
    
    Integrated with:
    - Limitation 3: Confidence-Gated Primal Diving & Sub-MIP Incumbent Discovery
    - Limitation 4: PyTorch INT8 Dynamic Quantization & Scale-Invariant Candidate Filtering
    
    At each branch-and-bound node:
    1. Computes candidate branching logits via the Bipartite GNN (optionally INT8-quantized).
    2. Candidate Pool Scale Control (Top-K filtering) if candidate set exceeds candidate_topk.
    3. Measures model confidence using:
       - Top-2 Softmax Margin: Marg = p_(1) - p_(2) in [0, 1]
       - Normalized Shannon Entropy: H_norm = H(p) / log(num_candidates) in [0, 1]
    4. Gate Rule:
       - If Margin >= margin_thresh and H_norm <= entropy_thresh:
           -> Branch using GNN (High Confidence)
       - Else:
           -> Fall back to SCIP's default Reliable Pseudocosts (Uncertain / High Entropy)
    """
    def __init__(self, gnn_model, margin_thresh=0.02, entropy_thresh=0.85, 
                 mode='confidence', random_fallback_prob=0.0, device='cpu',
                 candidate_topk=0, enable_primal_diving=False, dive_entropy_thresh=0.35,
                 dive_margin_thresh=0.25):
        self.model = gnn_model
        self.margin_thresh = margin_thresh
        self.entropy_thresh = entropy_thresh
        self.mode = mode
        self.random_fallback_prob = random_fallback_prob
        self.device = device
        self.candidate_topk = candidate_topk
        self.enable_primal_diving = enable_primal_diving
        self.dive_entropy_thresh = dive_entropy_thresh
        self.dive_margin_thresh = dive_margin_thresh

        self.reset_stats()

    def reset_stats(self):
        self.total_decisions = 0
        self.gnn_decisions = 0
        self.fallback_decisions = 0
        self.dive_triggers = 0
        self.norm_entropies = []
        self.raw_entropies = []
        self.margins = []

    def select_action(self, observation, action_set, pseudocosts):
        self.total_decisions += 1

        with torch.no_grad():
            c_feat = torch.from_numpy(observation.row_features.astype(np.float32)).to(self.device)
            e_idx = torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(self.device)
            e_attr = torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(self.device)
            v_feat = torch.from_numpy(observation.variable_features.astype(np.float32)).to(self.device)

            logits = self.model(c_feat, e_idx, e_attr, v_feat)
            
            # Scale-Invariant Candidate Pool Filtering (Limitation 4)
            if self.candidate_topk > 0 and len(action_set) > self.candidate_topk:
                # Pre-filter candidate pool by pseudocosts to bound candidate computation
                pseudo_sub = pseudocosts[action_set]
                topk_indices = np.argpartition(-pseudo_sub, self.candidate_topk)[:self.candidate_topk]
                action_set_active = action_set[topk_indices]
            else:
                action_set_active = action_set

            cand_logits = logits[action_set_active.astype(np.int64)]
            num_cands = len(cand_logits)

            probs = F.softmax(cand_logits, dim=-1)
            raw_entropy = (-probs * torch.log(probs + 1e-12)).sum().item()
            self.raw_entropies.append(raw_entropy)

            # Scale-invariant normalized entropy in [0, 1]
            max_entropy = float(np.log(max(2, num_cands)))
            norm_entropy = float(raw_entropy / max_entropy)
            self.norm_entropies.append(norm_entropy)

            if num_cands > 1:
                top2 = torch.topk(probs, 2).values
                margin = float((top2[0] - top2[1]).item())
            else:
                margin = 1.0
            self.margins.append(margin)

            # Check Primal Diving trigger (Limitation 3)
            if self.enable_primal_diving:
                if norm_entropy <= self.dive_entropy_thresh and margin >= self.dive_margin_thresh:
                    self.dive_triggers += 1

            # Decision gate:
            use_gnn = True
            if self.mode == 'confidence':
                is_confident = (margin >= self.margin_thresh) and (norm_entropy <= self.entropy_thresh)
                use_gnn = is_confident
            elif self.mode == 'random':
                use_gnn = (np.random.rand() >= self.random_fallback_prob)
            elif self.mode == 'always_gnn':
                use_gnn = True

            if use_gnn:
                self.gnn_decisions += 1
                chosen_action = action_set_active[cand_logits.argmax().item()]
            else:
                self.fallback_decisions += 1
                cand_pseudocosts = pseudocosts[action_set]
                chosen_action = action_set[cand_pseudocosts.argmax()]

            return chosen_action


def run_gated_evaluation(problem='setcover', n_instances=10, seed=0, time_limit=180,
                         margin_thresh=0.02, entropy_thresh=0.85, mode='confidence',
                         quantize_int8=False, candidate_topk=0,
                         enable_primal_diving=False, dive_freq=5,
                         instance_tier=None):
    device = 'cpu'
    model_path = f"model/{problem}/{seed}/train_params.pkl"
    print(f"Loading GNN model from {model_path}...")
    gnn_model = GNNPolicy().to(device)
    gnn_model.load_state_dict(torch.load(model_path, map_location=device))
    gnn_model.eval()

    if quantize_int8:
        print("[LIMITATION 4] Applying PyTorch INT8 Dynamic Quantization (Linear layers)...")
        gnn_model = torch.quantization.quantize_dynamic(gnn_model, {torch.nn.Linear}, dtype=torch.qint8)

    policy = ConfidenceGatedBranchingPolicy(
        gnn_model=gnn_model,
        margin_thresh=margin_thresh,
        entropy_thresh=entropy_thresh,
        mode=mode,
        device=device,
        candidate_topk=candidate_topk,
        enable_primal_diving=enable_primal_diving
    )

    instances = []
    if problem == 'setcover':
        if instance_tier is None or instance_tier == 'small':
            instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        if instance_tier is None or instance_tier == 'medium':
            instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        if instance_tier is None or instance_tier == 'big':
            instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]

    q_tag = "_int8" if quantize_int8 else "_fp32"
    d_tag = f"_dive{dive_freq}" if enable_primal_diving else ""
    k_tag = f"_topk{candidate_topk}" if candidate_topk > 0 else ""
    tier_tag = f"_{instance_tier}" if instance_tier else "_all"
    result_file = f"gated_eval_{problem}_{mode}{tier_tag}{q_tag}{d_tag}{k_tag}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', result_file)

    fieldnames = [
        'instance_type', 'instance_path', 'policy_mode',
        'quantize_int8', 'candidate_topk', 'enable_primal_diving', 'dive_triggers',
        'nnodes', 'nlps', 'stime', 'walltime', 'primal_bound', 'dual_bound', 'gap', 'status',
        'total_decisions', 'gnn_decisions', 'fallback_decisions', 'fallback_rate',
        'mean_norm_entropy', 'mean_raw_entropy', 'mean_margin'
    ]

    scip_parameters = {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        'limits/time': time_limit,
        'timing/clocktype': 1,
        'branching/vanillafullstrong/idempotent': True
    }

    if enable_primal_diving:
        print(f"[LIMITATION 3] Activating Sub-MIP Primal Diving Heuristics (freq={dive_freq})...")
        scip_parameters['heuristics/rens/freq'] = dive_freq
        scip_parameters['heuristics/rins/freq'] = dive_freq
        scip_parameters['heuristics/guideddiving/freq'] = dive_freq
        scip_parameters['heuristics/linesearchdiving/freq'] = dive_freq

    obs_functions = {
        'node_obs': ecole.observation.NodeBipartite(),
        'pseudocosts': ecole.observation.Pseudocosts()
    }

    print(f"\n=== Running Confidence-Gated Evaluation ({mode.upper()}) ===")
    print(f"INT8 Quantization: {quantize_int8} | Candidate Top-K: {candidate_topk} | Primal Diving: {enable_primal_diving}")
    print(f"Margin Threshold: {margin_thresh} | Normalized Entropy Threshold: {entropy_thresh}")
    print(f"Total instances to run: {len(instances)}")
    print(f"Saving results to: {csv_path}")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, inst in enumerate(instances, 1):
            inst_type = inst['type']
            inst_path = inst['path']
            print(f"\n[{idx}/{len(instances)}] [{inst_type.upper()}] {inst_path}")

            policy.reset_stats()
            env = ecole.environment.Branching(
                observation_function=obs_functions,
                scip_params=scip_parameters
            )
            env.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

            t0 = time.perf_counter()
            obs, act_set, _, done, _ = env.reset(inst_path)

            while not done:
                node_obs = obs['node_obs']
                pseudocosts = obs['pseudocosts']
                action = policy.select_action(node_obs, act_set, pseudocosts)
                obs, act_set, _, done, _ = env.step(action)

            walltime = time.perf_counter() - t0
            m = env.model.as_pyscipopt()

            fb_rate = (policy.fallback_decisions / max(1, policy.total_decisions)) * 100.0
            mean_norm_ent = float(np.mean(policy.norm_entropies)) if policy.norm_entropies else 0.0
            mean_raw_ent = float(np.mean(policy.raw_entropies)) if policy.raw_entropies else 0.0
            mean_marg = float(np.mean(policy.margins)) if policy.margins else 1.0

            rec = {
                'instance_type': inst_type,
                'instance_path': inst_path,
                'policy_mode': mode,
                'quantize_int8': quantize_int8,
                'candidate_topk': candidate_topk,
                'enable_primal_diving': enable_primal_diving,
                'dive_triggers': policy.dive_triggers,
                'nnodes': m.getNNodes(),
                'nlps': m.getNLPs(),
                'stime': m.getSolvingTime(),
                'walltime': walltime,
                'primal_bound': m.getPrimalbound(),
                'dual_bound': m.getDualbound(),
                'gap': m.getGap(),
                'status': m.getStatus(),
                'total_decisions': policy.total_decisions,
                'gnn_decisions': policy.gnn_decisions,
                'fallback_decisions': policy.fallback_decisions,
                'fallback_rate': fb_rate,
                'mean_norm_entropy': mean_norm_ent,
                'mean_raw_entropy': mean_raw_ent,
                'mean_margin': mean_marg
            }
            writer.writerow(rec)
            f.flush()

            print(f"   Nodes: {rec['nnodes']:4d} | Time: {walltime:.2f}s | Gap: {rec['gap']:.4f} | Primal: {rec['primal_bound']:.2f} | Dives: {policy.dive_triggers} | Status: {rec['status']}")

    print(f"\n=== Evaluation Complete: {csv_path} ===")
    return csv_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='setcover')
    parser.add_argument('--n-instances', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time-limit', type=int, default=180)
    parser.add_argument('--margin-thresh', type=float, default=0.02)
    parser.add_argument('--entropy-thresh', type=float, default=0.85)
    parser.add_argument('--mode', choices=['confidence', 'random', 'always_gnn'], default='confidence')
    parser.add_argument('--quantize-int8', action='store_true', help='Apply dynamic INT8 quantization to GNN linear layers')
    parser.add_argument('--candidate-topk', type=int, default=0, help='Max candidate pool size (0 = all candidates)')
    parser.add_argument('--enable-primal-diving', action='store_true', help='Enable Sub-MIP primal diving heuristics')
    parser.add_argument('--dive-freq', type=int, default=5, help='Frequency of Sub-MIP primal diving heuristics')
    parser.add_argument('--instance-tier', choices=['small', 'medium', 'big'], default=None, help='Filter to specific instance tier')
    args = parser.parse_args()

    run_gated_evaluation(
        problem=args.problem,
        n_instances=args.n_instances,
        seed=args.seed,
        time_limit=args.time_limit,
        margin_thresh=args.margin_thresh,
        entropy_thresh=args.entropy_thresh,
        mode=args.mode,
        quantize_int8=args.quantize_int8,
        candidate_topk=args.candidate_topk,
        enable_primal_diving=args.enable_primal_diving,
        dive_freq=args.dive_freq,
        instance_tier=args.instance_tier
    )

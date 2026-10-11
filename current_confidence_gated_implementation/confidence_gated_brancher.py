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

class AdaptiveMetaTriageController:
    """
    Learned/Heuristic Meta-Triage Controller to decide BEFORE running the GNN
    whether a branch-and-bound node can be safely resolved by SCIP's internal
    Reliable Pseudocosts.
    
    Triage Rules (evaluated in <1 microsecond):
    1. Single-candidate fast-path (|C| <= 1).
    2. Tree depth cutoff (depth > early_exit_depth).
    3. Fractionality Decisiveness: When LP solution feature shows one candidate
       deeply centered at 0.5 while all others are near bounds (margin >= frac_margin),
       the variable choice is trivially decisive on LP split impact.
    4. Pseudocost Decisiveness: At depth >= min_depth, if the top candidate's
       pseudocost has a decisive lead over the runner-up (relative margin >= pseudo_margin),
       SCIP's historical statistics are already decisive; running a 20ms GNN forward pass
       provides negligible marginal gain.
    """
    def __init__(self, early_exit_depth=20, pseudo_margin=0.35, min_depth=5, 
                 frac_margin=0.20, enabled=True):
        self.early_exit_depth = early_exit_depth
        self.pseudo_margin = pseudo_margin
        self.min_depth = min_depth
        self.frac_margin = frac_margin
        self.enabled = enabled

    def should_bypass_gnn(self, depth, pseudocosts, action_set, var_features=None):
        if not self.enabled:
            return False, "disabled"

        num_cands = len(action_set)
        if num_cands <= 1:
            return True, "trivial_cands"

        # 1. Depth-based cutoff:
        if self.early_exit_depth is not None and depth > self.early_exit_depth:
            return True, "depth_cutoff"

        # 2. Fractionality Decisiveness (LP Solution Topology):
        if var_features is not None and num_cands >= 2 and self.frac_margin is not None:
            cand_fracs = var_features[action_set.astype(np.int64), 8]
            frac_dist = np.abs(cand_fracs - 0.5)
            top2_idx = np.argpartition(frac_dist, 2)[:2]
            top2_dist = np.sort(frac_dist[top2_idx])
            d0, d1 = top2_dist[0], top2_dist[1]
            if d0 < 0.05 and (d1 - d0) >= self.frac_margin:
                return True, "fractionality_decisive"

        # 3. Adaptive Pseudocost Decisiveness:
        if depth >= self.min_depth:
            cand_pseudos = pseudocosts[action_set]
            if len(cand_pseudos) >= 2:
                top2_idx = np.argpartition(cand_pseudos, -2)[-2:]
                top2_vals = np.sort(cand_pseudos[top2_idx])[::-1]
                p1, p2 = top2_vals[0], top2_vals[1]
                if p1 > 1e-5:
                    rel_margin = (p1 - p2) / p1
                    if rel_margin >= self.pseudo_margin:
                        return True, "pseudocost_decisive"

        return False, "run_gnn"


class ConfidenceGatedBranchingPolicy:
    """
    Confidence-Gated Neural Branching Policy for MILP Solving.
    
    At each branch-and-bound node:
    1. Evaluates Adaptive Meta-Triage to check if GNN forward pass can be safely bypassed.
    2. If GNN is required:
       - Computes candidate logits with Cardinality-Aware Temperature Calibration & depth-aware scaling.
       - Evaluates Top-2 Softmax Margin (Marg) and Normalized Shannon Entropy (H_norm).
    3. Gate Rule:
       - If Margin >= margin_thresh and H_norm <= entropy_thresh:
           -> Branch using GNN (High Confidence)
       - Else:
           -> Fall back to SCIP's default Reliable Pseudocosts (Uncertain / High Entropy)
    """
    def __init__(self, gnn_model, margin_thresh=0.02, entropy_thresh=0.85, 
                 mode='confidence', random_fallback_prob=0.0, 
                 temperature=1.2, early_exit_depth=20, 
                 enable_triage=True, pseudo_margin=0.35, min_triage_depth=5,
                 frac_margin=0.20, cardinality_scaling=True,
                 device='cpu'):
        self.model = gnn_model
        self.margin_thresh = margin_thresh
        self.entropy_thresh = entropy_thresh
        self.mode = mode
        self.random_fallback_prob = random_fallback_prob
        self.temperature = max(1e-4, float(temperature))
        self.cardinality_scaling = cardinality_scaling
        self.device = device

        self.triage = AdaptiveMetaTriageController(
            early_exit_depth=early_exit_depth,
            pseudo_margin=pseudo_margin,
            min_depth=min_triage_depth,
            frac_margin=frac_margin,
            enabled=enable_triage
        )

        self.reset_stats()

    def reset_stats(self):
        self.total_decisions = 0
        self.gnn_decisions = 0
        self.fallback_decisions = 0
        self.early_exit_decisions = 0
        self.bypass_trivial_count = 0
        self.bypass_depth_count = 0
        self.bypass_pseudo_count = 0
        self.bypass_frac_count = 0
        self.norm_entropies = []
        self.raw_entropies = []
        self.margins = []

    def select_action(self, observation, action_set, pseudocosts, current_depth=0):
        self.total_decisions += 1
        num_cands = len(action_set)

        # 1. Evaluate Adaptive Meta-Triage Controller:
        bypass, reason = self.triage.should_bypass_gnn(
            current_depth, pseudocosts, action_set, var_features=observation.variable_features
        )
        if bypass:
            self.fallback_decisions += 1
            self.early_exit_decisions += 1
            if reason == "trivial_cands":
                self.bypass_trivial_count += 1
                return action_set[0]
            elif reason == "depth_cutoff":
                self.bypass_depth_count += 1
            elif reason == "pseudocost_decisive":
                self.bypass_pseudo_count += 1
            elif reason == "fractionality_decisive":
                self.bypass_frac_count += 1

            cand_pseudocosts = pseudocosts[action_set]
            return action_set[cand_pseudocosts.argmax()]

        # 2. GNN Inference with Cardinality-Aware Temperature Calibration:
        with torch.no_grad():
            c_feat = torch.from_numpy(observation.row_features.astype(np.float32)).to(self.device)
            e_idx = torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(self.device)
            e_attr = torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(self.device)
            v_feat = torch.from_numpy(observation.variable_features.astype(np.float32)).to(self.device)

            logits = self.model(c_feat, e_idx, e_attr, v_feat)
            cand_logits = logits[action_set.astype(np.int64)]

            # Effective temperature with cardinality-aware log-scale factor:
            # T_eff = T_0 * (ln(|C|) / ln(30))^0.5 * (1 + 0.005 * min(d, 40))
            if self.cardinality_scaling:
                ref_pool = 30.0
                card_factor = (float(np.log(max(2, num_cands))) / float(np.log(ref_pool))) ** 0.5
            else:
                card_factor = 1.0

            t_eff = self.temperature * card_factor * (1.0 + 0.005 * min(current_depth, 40))
            scaled_logits = cand_logits / max(1e-4, t_eff)
            probs = F.softmax(scaled_logits, dim=-1)

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
                chosen_action = action_set[cand_logits.argmax().item()]
            else:
                self.fallback_decisions += 1
                cand_pseudocosts = pseudocosts[action_set]
                chosen_action = action_set[cand_pseudocosts.argmax()]

            return chosen_action


def run_gated_evaluation(problem='setcover', n_instances=10, seed=0, time_limit=180,
                         margin_thresh=0.02, entropy_thresh=0.85, mode='confidence',
                         temperature=1.2, early_exit_depth=20, enable_triage=True,
                         pseudo_margin=0.35, frac_margin=0.20, cardinality_scaling=True,
                         tiers='small,medium,big'):
    device = 'cpu'
    model_path = f"model/{problem}/{seed}/train_params.pkl"
    print(f"Loading GNN model from {model_path}...")
    gnn_model = GNNPolicy().to(device)
    gnn_model.load_state_dict(torch.load(model_path, map_location=device))
    gnn_model.eval()

    policy = ConfidenceGatedBranchingPolicy(
        gnn_model=gnn_model,
        margin_thresh=margin_thresh,
        entropy_thresh=entropy_thresh,
        mode=mode,
        temperature=temperature,
        early_exit_depth=early_exit_depth,
        enable_triage=enable_triage,
        pseudo_margin=pseudo_margin,
        frac_margin=frac_margin,
        cardinality_scaling=cardinality_scaling,
        device=device
    )

    selected_tiers = [t.strip().lower() for t in tiers.split(',')]
    instances = []
    if problem == 'setcover':
        if 'small' in selected_tiers:
            instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        if 'medium' in selected_tiers:
            instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        if 'big' in selected_tiers:
            instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]

    exit_str = f"_d{early_exit_depth}" if early_exit_depth is not None else ""
    temp_str = f"_T{temperature}" if temperature != 1.0 else ""
    triage_str = f"_pm{pseudo_margin}_fm{frac_margin}" if enable_triage else "_notriage"
    result_file = f"gated_eval_{problem}_{mode}_m{margin_thresh}_h{entropy_thresh}{temp_str}{exit_str}{triage_str}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', result_file)

    fieldnames = [
        'instance_type', 'instance_path', 'policy_mode',
        'temperature', 'early_exit_depth', 'pseudo_margin', 'frac_margin',
        'nnodes', 'nlps', 'stime', 'walltime', 'gap', 'status',
        'total_decisions', 'gnn_decisions', 'fallback_decisions', 'fallback_rate',
        'early_exit_decisions', 'early_exit_rate',
        'bypass_trivial', 'bypass_depth', 'bypass_pseudo', 'bypass_frac',
        'mean_norm_entropy', 'mean_raw_entropy', 'mean_margin'
    ]

    scip_parameters = {
        'separating/maxrounds': 0,
        'presolving/maxrestarts': 0,
        'limits/time': time_limit,
        'timing/clocktype': 1,
        'branching/vanillafullstrong/idempotent': True
    }

    obs_functions = {
        'node_obs': ecole.observation.NodeBipartite(),
        'pseudocosts': ecole.observation.Pseudocosts()
    }

    print(f"\n=== Running Confidence-Gated Evaluation ({mode.upper()}) ===")
    print(f"Margin: {margin_thresh} | Norm-H: {entropy_thresh} | Temp: {temperature} (CardScaling: {cardinality_scaling})")
    print(f"Adaptive Triage: {enable_triage} | Depth Cutoff: {early_exit_depth} | Pseudo Margin: {pseudo_margin} | Frac Margin: {frac_margin}")
    print(f"Total instances: {len(instances)} (Tiers: {tiers})")
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
                curr_depth = env.model.as_pyscipopt().getDepth()
                action = policy.select_action(node_obs, act_set, pseudocosts, current_depth=curr_depth)
                obs, act_set, _, done, _ = env.step(action)

            walltime = time.perf_counter() - t0
            m = env.model.as_pyscipopt()

            fb_rate = (policy.fallback_decisions / max(1, policy.total_decisions)) * 100.0
            ee_rate = (policy.early_exit_decisions / max(1, policy.total_decisions)) * 100.0
            mean_norm_ent = float(np.mean(policy.norm_entropies)) if policy.norm_entropies else 0.0
            mean_raw_ent = float(np.mean(policy.raw_entropies)) if policy.raw_entropies else 0.0
            mean_marg = float(np.mean(policy.margins)) if policy.margins else 1.0

            rec = {
                'instance_type': inst_type,
                'instance_path': inst_path,
                'policy_mode': mode,
                'temperature': policy.temperature,
                'early_exit_depth': policy.triage.early_exit_depth,
                'pseudo_margin': policy.triage.pseudo_margin,
                'frac_margin': policy.triage.frac_margin,
                'nnodes': m.getNNodes(),
                'nlps': m.getNLPs(),
                'stime': m.getSolvingTime(),
                'walltime': walltime,
                'gap': m.getGap(),
                'status': m.getStatus(),
                'total_decisions': policy.total_decisions,
                'gnn_decisions': policy.gnn_decisions,
                'fallback_decisions': policy.fallback_decisions,
                'fallback_rate': fb_rate,
                'early_exit_decisions': policy.early_exit_decisions,
                'early_exit_rate': ee_rate,
                'bypass_trivial': policy.bypass_trivial_count,
                'bypass_depth': policy.bypass_depth_count,
                'bypass_pseudo': policy.bypass_pseudo_count,
                'bypass_frac': policy.bypass_frac_count,
                'mean_norm_entropy': mean_norm_ent,
                'mean_raw_entropy': mean_raw_ent,
                'mean_margin': mean_marg
            }
            writer.writerow(rec)
            f.flush()

            print(f"   Nodes: {rec['nnodes']:4d} | Time: {walltime:.2f}s | Status: {rec['status']} | Fallback: {fb_rate:.1f}% (Bypass: {ee_rate:.1f}% [depth={policy.bypass_depth_count}, pseudo={policy.bypass_pseudo_count}, frac={policy.bypass_frac_count}]) | Norm-H: {mean_norm_ent:.2f}")

    print(f"\n=== Evaluation Complete: {csv_path} ===")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='setcover')
    parser.add_argument('--n-instances', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--time-limit', type=int, default=180)
    parser.add_argument('--margin-thresh', type=float, default=0.02)
    parser.add_argument('--entropy-thresh', type=float, default=0.85)
    parser.add_argument('--temperature', type=float, default=1.2, help='Softmax temperature scaling parameter')
    parser.add_argument('--early-exit-depth', type=int, default=20, help='Max B&B depth to run GNN before depth-cutoff bypass')
    parser.add_argument('--no-triage', action='store_true', help='Disable adaptive meta-triage')
    parser.add_argument('--pseudo-margin', type=float, default=0.35, help='Relative pseudocost margin to trigger fast bypass')
    parser.add_argument('--frac-margin', type=float, default=0.20, help='Fractionality distance separation to trigger fast bypass')
    parser.add_argument('--no-cardinality-scaling', action='store_true', help='Disable log-cardinality temperature scaling')
    parser.add_argument('--tiers', default='small,medium,big', help='Comma-separated tiers to evaluate')
    parser.add_argument('--mode', choices=['confidence', 'random', 'always_gnn'], default='confidence')
    args = parser.parse_args()

    run_gated_evaluation(
        problem=args.problem,
        n_instances=args.n_instances,
        seed=args.seed,
        time_limit=args.time_limit,
        margin_thresh=args.margin_thresh,
        entropy_thresh=args.entropy_thresh,
        mode=args.mode,
        temperature=args.temperature,
        early_exit_depth=args.early_exit_depth,
        enable_triage=not args.no_triage,
        pseudo_margin=args.pseudo_margin,
        frac_margin=args.frac_margin,
        cardinality_scaling=not args.no_cardinality_scaling,
        tiers=args.tiers
    )

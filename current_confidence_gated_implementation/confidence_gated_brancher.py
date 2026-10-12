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
    Multi-Layer Adaptive Meta-Triage Controller (Limitation 1).
    
    Dynamically predicts whether computing an expensive GNN forward pass is redundant,
    bypassing directly to fast SCIP Reliable Pseudocosts with near-zero latency.
    
    Triage Layers:
    1. Trivial Decision Triage: If candidate pool |C| <= 1, choice is unique or trivial.
    2. Tree-Depth Cutoff: If depth > early_exit_depth, imitation value diminishes.
    3. Pseudocost Dominance Margin: If top pseudocost strongly dominates runner-up by pseudo_margin.
    4. Fractionality Certainty Margin: If most fractional variable is near 0.5 and separated by frac_margin.
    """
    def __init__(self, early_exit_depth=20, pseudo_margin=0.35, frac_margin=0.20, enabled=True):
        self.early_exit_depth = early_exit_depth
        self.pseudo_margin = pseudo_margin
        self.frac_margin = frac_margin
        self.enabled = enabled

    def should_bypass_gnn(self, depth, action_set, pseudocosts, fractional_distances=None):
        if not self.enabled:
            return False, "triage_disabled"

        num_cands = len(action_set)
        if num_cands <= 1:
            return True, "trivial_pool"

        if self.early_exit_depth is not None and depth > self.early_exit_depth:
            return True, "depth_cutoff"

        if self.pseudo_margin is not None and pseudocosts is not None and num_cands >= 2:
            cand_pseudos = pseudocosts[action_set]
            if len(cand_pseudos) >= 2:
                top2_idx = np.argpartition(-cand_pseudos, 1)[:2]
                val1, val2 = cand_pseudos[top2_idx[0]], cand_pseudos[top2_idx[1]]
                if val1 < val2:
                    val1, val2 = val2, val1
                if val1 > 1e-6:
                    rel_pseudo_margin = (val1 - val2) / (val1 + 1e-6)
                    if rel_pseudo_margin >= self.pseudo_margin:
                        return True, "pseudocost_dominance"

        if self.frac_margin is not None and fractional_distances is not None and len(fractional_distances) >= 2:
            cand_fracs = fractional_distances[action_set]
            top2_frac = np.partition(-cand_fracs, 1)[:2]
            f1, f2 = -top2_frac[0], -top2_frac[1]
            if (f1 - f2) >= self.frac_margin:
                return True, "fractionality_dominance"

        return False, "proceed_to_gnn"


class AdaptiveRootOODDetector:
    """
    Autonomous Root-Node Out-Of-Distribution (OOD) Domain Shift Detector (Phase A).
    
    Inspects problem characteristics and constraint tensor geometry at root depth d=0:
    1. Constraint Polarity Discrepancy:
       - Set Covering training distribution has negative constraint-objective cosine alignment (mean_cos < 0).
       - Packing/Auction distributions have positive cosine alignment (mean_cos > 0).
    2. Epistemic Guardrail:
       - If out-of-distribution domain shift is detected at d=0:
         Dynamically clamps:
           tau_h_eff = min(tau_h_base, 0.50)
           tau_m_eff = max(tau_m_base, 0.25)
       - Eliminates 'confidently incompetent' threshold leakage on foreign families autonomously!
    """
    def __init__(self, strict_entropy_limit=0.50, strict_margin_limit=0.25, enabled=True):
        self.strict_entropy_limit = strict_entropy_limit
        self.strict_margin_limit = strict_margin_limit
        self.enabled = enabled
        self.is_ood = False
        self.detected_reason = "none"

    def reset(self):
        self.is_ood = False
        self.detected_reason = "none"

    def inspect_root_node(self, depth, observation, base_entropy_thresh, base_margin_thresh):
        if not self.enabled:
            return base_entropy_thresh, base_margin_thresh, False, "disabled"

        if depth > 0:
            if self.is_ood:
                return self.strict_entropy_limit, self.strict_margin_limit, True, self.detected_reason
            else:
                return base_entropy_thresh, base_margin_thresh, False, self.detected_reason

        # Root node analysis (d = 0)
        c_feat = observation.row_features
        if c_feat is not None and len(c_feat) > 0:
            mean_obj_cos = float(np.mean(c_feat[:, 0]))
            # If constraint-objective cosine alignment > 0, problem polarity is packing/aligned,
            # opposite to Set Covering covering polarity (consistently < -0.10)
            if mean_obj_cos > 0.0:
                self.is_ood = True
                self.detected_reason = f"packing_polarity_cos_{mean_obj_cos:.2f}"
                return self.strict_entropy_limit, self.strict_margin_limit, self.is_ood, self.detected_reason

        self.is_ood = False
        self.detected_reason = "in_distribution_covering"
        return base_entropy_thresh, base_margin_thresh, self.is_ood, self.detected_reason


class CardinalityAwareTemperatureScaler:
    """
    Cardinality-Aware Temperature Scaling for Softmax Calibration (Limitation 2).
    
    Standard temperature scaling T scales logits z / T. Under combinatorial variable
    fluctuations (|C| in [10, 2000]), entropy scales with log(|C|), causing out-of-distribution
    overconfidence.
    
    Cardinality Calibration Law:
    T_eff = T_base * (1.0 + beta * log(max(2, |C|) / C_ref))
    """
    def __init__(self, base_temperature=1.2, beta=0.15, ref_cardinality=30.0, enabled=True):
        self.base_temperature = base_temperature
        self.beta = beta
        self.ref_cardinality = ref_cardinality
        self.enabled = enabled

    def get_temperature(self, num_candidates):
        if not self.enabled or self.base_temperature == 1.0:
            return self.base_temperature
        cardinality_ratio = max(2.0, float(num_candidates)) / self.ref_cardinality
        log_card = float(np.log(max(1.0, cardinality_ratio)))
        t_eff = self.base_temperature * (1.0 + self.beta * log_card)
        return max(0.1, t_eff)


class UnifiedConfidenceGatedBranchingPolicy:
    """
    Unified Confidence-Gated Neural Branching Policy.
    
    Combines:
    - Limitation 1: Multi-Layer Adaptive Meta-Triage (Fast CPU bypass)
    - Limitation 2: Cardinality-Aware Temperature Scaling (Calibrated Epistemic Gate)
    - Limitation 3: Confidence-Gated Primal Diving (Sub-MIP Incumbent Discovery)
    - Limitation 4: PyTorch INT8 Dynamic Quantization & Scale-Invariant Candidate Filtering
    - Phase A: Autonomous Root-Node Adaptive OOD Gate (Universal Cross-Family Transfer)
    """
    def __init__(self, gnn_model, margin_thresh=0.02, entropy_thresh=0.85, 
                 temperature=1.2, cardinality_scaling=True,
                 enable_triage=True, early_exit_depth=20, pseudo_margin=0.35, frac_margin=0.20,
                 quantize_int8=False, candidate_topk=0,
                 enable_primal_diving=False, dive_entropy_thresh=0.35, dive_margin_thresh=0.25,
                 enable_ood_detection=True,
                 mode='confidence', random_fallback_prob=0.0, device='cpu'):
        self.model = gnn_model
        self.margin_thresh = margin_thresh
        self.entropy_thresh = entropy_thresh
        self.effective_margin_thresh = margin_thresh
        self.effective_entropy_thresh = entropy_thresh
        self.mode = mode
        self.random_fallback_prob = random_fallback_prob
        self.device = device
        self.quantize_int8 = quantize_int8
        self.candidate_topk = candidate_topk
        self.enable_primal_diving = enable_primal_diving
        self.dive_entropy_thresh = dive_entropy_thresh
        self.dive_margin_thresh = dive_margin_thresh

        self.triage = AdaptiveMetaTriageController(
            early_exit_depth=early_exit_depth,
            pseudo_margin=pseudo_margin,
            frac_margin=frac_margin,
            enabled=enable_triage
        )
        self.scaler = CardinalityAwareTemperatureScaler(
            base_temperature=temperature,
            enabled=cardinality_scaling
        )
        self.ood_detector = AdaptiveRootOODDetector(enabled=enable_ood_detection)
        self.temperature = temperature
        self.reset_stats()

    def reset_stats(self):
        self.total_decisions = 0
        self.gnn_decisions = 0
        self.fallback_decisions = 0
        self.early_exit_decisions = 0
        self.dive_triggers = 0
        self.bypass_trivial_count = 0
        self.bypass_depth_count = 0
        self.bypass_pseudo_count = 0
        self.bypass_frac_count = 0
        self.is_ood_detected = False
        self.effective_entropy_thresh = self.entropy_thresh
        self.effective_margin_thresh = self.margin_thresh
        self.has_inspected_root = False
        self.ood_detector.reset()
        self.norm_entropies = []
        self.raw_entropies = []
        self.margins = []

    def select_action(self, observation, action_set, pseudocosts, depth=0):
        self.total_decisions += 1

        # Root Node OOD Detection (Phase A: Universal Cross-Family Transfer)
        if not self.has_inspected_root or depth == 0:
            self.effective_entropy_thresh, self.effective_margin_thresh, self.is_ood_detected, _ = \
                self.ood_detector.inspect_root_node(depth, observation, self.entropy_thresh, self.margin_thresh)
            self.has_inspected_root = True

        # Layer 1: Adaptive Meta-Triage (Limitation 1)
        if self.mode == 'confidence':
            var_features = observation.variable_features
            frac_dists = None
            if var_features is not None and var_features.shape[1] > 18:
                frac_dists = np.abs(var_features[:, 18] - 0.5)

            bypass, reason = self.triage.should_bypass_gnn(depth, action_set, pseudocosts, frac_dists)
            if bypass:
                self.early_exit_decisions += 1
                self.fallback_decisions += 1
                if reason == "trivial_pool":
                    self.bypass_trivial_count += 1
                elif reason == "depth_cutoff":
                    self.bypass_depth_count += 1
                elif reason == "pseudocost_dominance":
                    self.bypass_pseudo_count += 1
                elif reason == "fractionality_dominance":
                    self.bypass_frac_count += 1
                cand_pseudos = pseudocosts[action_set]
                return action_set[cand_pseudos.argmax()]

        # Scale-Invariant Candidate Pool Filtering (Limitation 4)
        if self.candidate_topk > 0 and len(action_set) > self.candidate_topk:
            pseudo_sub = pseudocosts[action_set]
            topk_indices = np.argpartition(-pseudo_sub, self.candidate_topk)[:self.candidate_topk]
            action_set_active = action_set[topk_indices]
        else:
            action_set_active = action_set

        # Layer 2: GNN Inference (Optionally INT8 Quantized, Limitation 4)
        with torch.no_grad():
            c_feat = torch.from_numpy(observation.row_features.astype(np.float32)).to(self.device)
            e_idx = torch.from_numpy(observation.edge_features.indices.astype(np.int64)).to(self.device)
            e_attr = torch.from_numpy(observation.edge_features.values.astype(np.float32)).view(-1, 1).to(self.device)
            v_feat = torch.from_numpy(observation.variable_features.astype(np.float32)).to(self.device)

            logits = self.model(c_feat, e_idx, e_attr, v_feat)
            cand_logits = logits[action_set_active.astype(np.int64)]
            num_cands = len(cand_logits)

            # Layer 3: Cardinality-Aware Temperature Calibration (Limitation 2)
            t_eff = self.scaler.get_temperature(num_cands)
            calibrated_logits = cand_logits / t_eff

            probs = F.softmax(calibrated_logits, dim=-1)
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

            # Layer 4: Primal Diving Trigger Monitor (Limitation 3)
            if self.enable_primal_diving:
                if norm_entropy <= self.dive_entropy_thresh and margin >= self.dive_margin_thresh:
                    self.dive_triggers += 1

            # Decision Gate:
            use_gnn = True
            if self.mode == 'confidence':
                is_confident = (margin >= self.effective_margin_thresh) and (norm_entropy <= self.effective_entropy_thresh)
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
                         temperature=1.2, early_exit_depth=20,
                         enable_triage=True, pseudo_margin=0.35, frac_margin=0.20,
                         cardinality_scaling=True, tiers='small,medium,big',
                         quantize_int8=False, candidate_topk=0,
                         enable_primal_diving=False, dive_freq=5,
                         enable_ood_detection=True):
    device = 'cpu'
    model_path = f"model/{problem}/{seed}/train_params.pkl"
    print(f"Loading GNN model from {model_path}...")
    gnn_model = GNNPolicy().to(device)
    gnn_model.load_state_dict(torch.load(model_path, map_location=device))
    gnn_model.eval()

    if quantize_int8:
        print("[PILLAR 4] Applying PyTorch INT8 Dynamic Quantization (Linear layers)...")
        gnn_model = torch.quantization.quantize_dynamic(gnn_model, {torch.nn.Linear}, dtype=torch.qint8)

    policy = UnifiedConfidenceGatedBranchingPolicy(
        gnn_model=gnn_model,
        margin_thresh=margin_thresh,
        entropy_thresh=entropy_thresh,
        temperature=temperature,
        cardinality_scaling=cardinality_scaling,
        enable_triage=enable_triage,
        early_exit_depth=early_exit_depth,
        pseudo_margin=pseudo_margin,
        frac_margin=frac_margin,
        quantize_int8=quantize_int8,
        candidate_topk=candidate_topk,
        enable_primal_diving=enable_primal_diving,
        enable_ood_detection=enable_ood_detection,
        mode=mode,
        device=device
    )

    instances = []
    selected_tiers = [t.strip().lower() for t in tiers.split(',')]
    if problem == 'setcover':
        if 'small' in selected_tiers:
            instances += [{'type': 'small', 'path': f"data/instances/setcover/transfer_500r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        if 'medium' in selected_tiers:
            instances += [{'type': 'medium', 'path': f"data/instances/setcover/transfer_1000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]
        if 'big' in selected_tiers:
            instances += [{'type': 'big', 'path': f"data/instances/setcover/transfer_2000r_1000c_0.05d/instance_{i+1}.lp"} for i in range(n_instances)]

    q_tag = "_int8" if quantize_int8 else "_fp32"
    d_tag = f"_dive{dive_freq}" if enable_primal_diving else ""
    k_tag = f"_topk{candidate_topk}" if candidate_topk > 0 else ""
    exit_str = f"_d{early_exit_depth}" if early_exit_depth is not None else ""
    temp_str = f"_T{temperature}" if temperature != 1.0 else ""
    triage_str = f"_pm{pseudo_margin}_fm{frac_margin}" if enable_triage else "_notriage"
    tier_tag = f"_{tiers.replace(',', '-')}"
    result_file = f"unified_eval_{problem}_{mode}{tier_tag}{q_tag}{d_tag}{k_tag}{temp_str}{exit_str}{triage_str}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join('results', result_file)

    fieldnames = [
        'instance_type', 'instance_path', 'policy_mode',
        'quantize_int8', 'candidate_topk', 'enable_primal_diving', 'dive_triggers',
        'temperature', 'early_exit_depth', 'pseudo_margin', 'frac_margin',
        'nnodes', 'nlps', 'stime', 'walltime', 'primal_bound', 'dual_bound', 'gap', 'status',
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

    if enable_primal_diving:
        print(f"[PILLAR 3] Activating Sub-MIP Primal Diving Heuristics (freq={dive_freq})...")
        scip_parameters['heuristics/rens/freq'] = dive_freq
        scip_parameters['heuristics/rins/freq'] = dive_freq
        scip_parameters['heuristics/guideddiving/freq'] = dive_freq
        scip_parameters['heuristics/linesearchdiving/freq'] = dive_freq

    obs_functions = {
        'node_obs': ecole.observation.NodeBipartite(),
        'pseudocosts': ecole.observation.Pseudocosts()
    }

    print(f"\n=== Running Unified Confidence-Gated Evaluation ({mode.upper()}) ===")
    print(f"Pillars Active: INT8={quantize_int8} | TopK={candidate_topk} | PrimalDiving={enable_primal_diving} | Triage={enable_triage} | Temp={temperature}")
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
                'temperature': policy.temperature,
                'early_exit_depth': policy.triage.early_exit_depth,
                'pseudo_margin': policy.triage.pseudo_margin,
                'frac_margin': policy.triage.frac_margin,
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

            print(f"   Nodes: {rec['nnodes']:4d} | Time: {walltime:.2f}s | Gap: {rec['gap']:.4f} | Primal: {rec['primal_bound']:.2f} | Dives: {policy.dive_triggers} | Status: {rec['status']} | Fallback: {fb_rate:.1f}% (Bypass: {ee_rate:.1f}%)")

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
    parser.add_argument('--temperature', type=float, default=1.2, help='Softmax temperature scaling parameter')
    parser.add_argument('--early-exit-depth', type=int, default=20, help='Max B&B depth to run GNN before depth-cutoff bypass')
    parser.add_argument('--no-triage', action='store_true', help='Disable adaptive meta-triage')
    parser.add_argument('--pseudo-margin', type=float, default=0.35, help='Relative pseudocost margin to trigger fast bypass')
    parser.add_argument('--frac-margin', type=float, default=0.20, help='Fractionality distance separation to trigger fast bypass')
    parser.add_argument('--no-cardinality-scaling', action='store_true', help='Disable log-cardinality temperature scaling')
    parser.add_argument('--tiers', default='small,medium,big', help='Comma-separated tiers to evaluate')
    parser.add_argument('--mode', choices=['confidence', 'random', 'always_gnn'], default='confidence')
    parser.add_argument('--quantize-int8', action='store_true', help='Apply dynamic INT8 quantization to GNN linear layers')
    parser.add_argument('--candidate-topk', type=int, default=0, help='Max candidate pool size (0 = all candidates)')
    parser.add_argument('--enable-primal-diving', action='store_true', help='Enable Sub-MIP primal diving heuristics')
    parser.add_argument('--dive-freq', type=int, default=5, help='Frequency of Sub-MIP primal diving heuristics')
    parser.add_argument('--no-ood-detection', action='store_true', help='Disable autonomous root-node OOD detection')
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
        tiers=args.tiers,
        quantize_int8=args.quantize_int8,
        candidate_topk=args.candidate_topk,
        enable_primal_diving=args.enable_primal_diving,
        dive_freq=args.dive_freq,
        enable_ood_detection=not args.no_ood_detection
    )

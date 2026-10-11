# Confidence-Gated Neural Branching for Robust MILP Solving

This repository contains the standalone, reproducible implementation of **Confidence-Gated Neural Branching for Robust Mixed-Integer Linear Programming (MILP)**. 

By augmenting Graph Convolutional Neural Networks (GCNNs) with scale-invariant epistemic uncertainty gating, the solver selectively branches with neural predictions when confident, and safely defers to SCIP's internal **Reliable Pseudocosts (`relpscost`)** when uncertain or out-of-distribution.

---

## 1. Directory Structure

```text
current_confidence_gated_implementation/
├── confidence_gated_brancher.py    # Core implementation: Confidence-Gated Brancher with Normalized Entropy & Margin
├── eval_benchmark.py               # Benchmark runner: SCIP default vs. Pure GNN baseline
├── model/
│   ├── model.py                    # Bipartite Graph Convolutional Neural Network (GCNN)
│   └── setcover/                   # Pretrained model weights
│       ├── 0/train_params.pkl      # Seed 0 checkpoint
│       └── 1/train_params.pkl      # Seed 1 checkpoint
├── results/                        # Full evaluation CSV logs across Small, Medium, Big instances
│   ├── eval_setcover_10inst_seed0_*.csv
│   └── gated_eval_setcover_confidence_*.csv
├── utilities.py                    # Graph data loaders, schedulers, and helper functions
├── RESEARCH_COMPENDIUM.md          # Authoritative Scientific Compendium (Theory, Literature, Full Logs, BibTeX)
└── README.md                       # This quick-start and operational guide
```

---

## 2. Core Method at a Glance

At each branch-and-bound node with candidate set $\mathcal{C}$ and softmax distribution $p$:

1. **Scale-Invariant Normalized Shannon Entropy ($\tilde{H} \in [0, 1]$):**
   $$\tilde{H}(p) = \frac{-\sum_{c \in \mathcal{C}} p_c \log p_c}{\log(\max(2, |\mathcal{C}|))}$$
   *Normalizes entropy against candidate pool cardinality $|\mathcal{C}|$, enabling a single universal threshold across problem sizes.*

2. **Top-2 Softmax Margin ($\Delta p \in [0, 1]$):**
   $$\Delta p = p_{(1)} - p_{(2)}$$
   *Measures decision decisiveness between the top contender and runner-up.*

3. **Dual-Gating Decision Rule:**
   - **Execute GNN Branching:** If $\Delta p \ge \tau_m$ ($0.02$) **AND** $\tilde{H} \le \tau_h$ ($0.85$).
   - **Fallback to SCIP Pseudocosts:** Otherwise (instant, zero-overhead mathematical safety net).

---

## 3. Quick-Start: Running via Docker

All evaluations run deterministically inside the provided Docker environment (`confidence-branching:latest`).

### A. Run the Confidence-Gated Brancher:
```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace/current_confidence_gated_implementation \
  confidence-branching python confidence_gated_brancher.py \
  --problem setcover \
  --n-instances 10 \
  --margin-thresh 0.02 \
  --entropy-thresh 0.85 \
  --mode confidence \
  --time-limit 180
```

### B. Run the Baseline Benchmark (SCIP Default vs. Pure GNN):
```bash
docker run --rm -v "${PWD}:/workspace" -w /workspace/current_confidence_gated_implementation \
  confidence-branching python eval_benchmark.py \
  --problem setcover \
  --n-instances 10 \
  --seed 0 \
  --time-limit 180
```

### C. Available CLI Modes:
- `--mode confidence`: Runs the proposed Dual Confidence-Gated Brancher.
- `--mode pure_gnn`: Runs the un-gated pure GNN policy (unconditional argmax).
- `--mode pure_pseudocost`: Runs pure pseudocost heuristic for ablation.

---

## 4. Key Benchmark Results

Validated on 30 benchmark instances across three scale tiers (Set Covering instances with $180\text{s}$ time limit):

| Instance Tier | Problem Size ($m \times n$) | SCIP Default Solved | Pure GNN Solved | Gated (Ours) Solved | Key Metric Highlight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Small** | $500 \times 1,000$ | 10 / 10 | 8 / 10 | **10 / 10** | Restored 100% solve rate; 34% median speedup over Pure GNN |
| **Medium** | $1,000 \times 1,000$ | 9 / 10 | 8 / 10 | **10 / 10** | Solved hard instances to proven optimality in 25s |
| **Big ($4\times$ shift)** | $2,000 \times 1,000$ | 0 / 10 (Timeouts) | 0 / 10 (Timeouts) | **Pareto Dominant** | **$63\times$ tree node reduction** (mean 118.5 vs. 7,444.8 nodes) |

*Note: Node counts reported are arithmetic means across the 10 instances in the tier (total 1,185 vs. 74,448 nodes).*

---

## 5. Authoritative Scientific Treatise: RESEARCH_COMPENDIUM.md

To maintain a clean separation between operational usage and scientific research, all detailed mathematical derivations, literature comparisons, and per-instance raw logs are maintained in [RESEARCH_COMPENDIUM.md](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/gnn/Confidence-Gated-Neural-Branching-for-Robust-MILP-Solving/current_confidence_gated_implementation/RESEARCH_COMPENDIUM.md).

Please refer to [RESEARCH_COMPENDIUM.md](file:///c:/Users/likhi/OneDrive/Pictures/Desktop/gnn/Confidence-Gated-Neural-Branching-for-Robust-MILP-Solving/current_confidence_gated_implementation/RESEARCH_COMPENDIUM.md) for:
1. **Mathematical & Graph Formulations:** Lossless bipartite graph formulation $G = (\mathcal{V}_{var} \cup \mathcal{V}_{con}, \mathcal{E})$ and LP relaxation embeddings.
2. **Exhaustive Literature Breakdown:** Detailed attribution comparing what prior papers established (Gasse et al. 2019, Nair et al. 2020, Gupta et al. 2020, Ross et al. 2011) versus what was directly measured in our empirical benchmarks.
3. **Foundational Academic Lineage:** Theoretical foundations for our novel components (Shannon 1948, Pielou 1966, Settles 2009, Joshi et al. 2009, Chow 1970, Cortes et al. 2016, Madras et al. 2018).
4. **Complete Per-Instance Execution Logs:** Full runtime, node count, and fallback percentage logs for all 30 evaluated instances.
5. **Scientific Limitations & Proposed Mitigations:** In-depth discussion of forward-pass sunk costs, calibration vulnerabilities, and primal/dual asymmetry.
6. **Academic References & BibTeX:** Full bibliography for academic citations and thesis integration.

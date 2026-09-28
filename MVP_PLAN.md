# MVP Implementation Plan

## Objective

Test whether GNN confidence predicts poor MILP branching decisions and whether low-confidence decisions should fall back to classical branching.

## Fixed MVP scope

- Base model: existing bipartite GCNN from `repos/learn2branch-ecole`
- Confidence signals: softmax margin and entropy
- MILP families: set covering, combinatorial auction, capacitated facility location, maximum independent set
- Splits: in-distribution, size shift, structural shift
- Conditions: always-GNN, always-classical, confidence-gated, matched-rate random fallback
- Decision metrics: strong-branching agreement, rank, score, regret, bound improvement
- Solver metrics: primal-dual integral, runtime, time to target gap, solved instances, final gap, nodes, overhead

## Stages

1. Audit the repositories and environment.
2. Reproduce the base GNN on a small set-covering experiment.
3. Establish always-GNN and always-classical baselines.
4. Log per-node probabilities, margin, entropy, and decision quality.
5. Analyze RQ1: confidence versus branching quality.
6. Implement confidence gates using validation-only thresholds.
7. Implement matched-rate random fallback.
8. Run controlled comparisons across families and shifts.
9. Analyze overhead and confidently-wrong decisions.
10. Write conclusions for RQ1, RQ2, and RQ3.

## First milestone

Run `learn2branch-ecole` on a small set-covering dataset and document the exact environment, commands, model output, and solver result before modifying the brancher.

## Stop/go rule

Only build the gate if margin or entropy shows useful predictive behavior in the RQ1 analysis. A negative RQ1 result is a valid project result.

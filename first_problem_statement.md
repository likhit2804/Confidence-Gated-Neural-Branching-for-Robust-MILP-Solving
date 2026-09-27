Confidence-Gated Neural Branching: Can a Learned Policy Recognize When It Should Not Be Trusted?

Revised to separate the prediction and intervention questions, avoid assuming causality or negligible overhead, and define a minimum viable experiment before optional extensions.

Abstract

Learned branching policies for mixed-integer linear programming (MILP) can provide computational advantages on some instance distributions but degrade substantially under distributional shift. Existing evaluations compare learned and classical branching policies as fixed, always-on strategies, leaving open whether unreliable learned decisions can be identified at inference time, before they affect the search tree. This project investigates whether the output distribution of a bipartite GNN branching policy provides a useful per-node signal of branching-decision reliability. It separates this into two questions: whether confidence predicts decision quality (a correlational question), and whether acting on that signal — falling back to a classical rule on low-confidence decisions — improves solver-level performance (an interventional question). A third question isolates whether any benefit comes from the signal itself or merely from occasionally using the classical rule. The project is scoped to a minimum viable experiment (one architecture, two cheap confidence signals, four established MILP families) before any optional extension.

1. Background and Related Work Framing

This proposal builds on three strands of prior work.

Methodological framing. The general case for learning inside a CO solver, rather than learning an end-to-end answer, rests on preserving the solver's feasibility and search guarantees while replacing an expensive expert subroutine with a fast approximation. The recurring caution is generalization: because branching decisions affect the states visited later, an early error can compound, and a policy trained on one distribution or one region of a search trajectory can behave unpredictably elsewhere. This motivates evaluating any learned component across held-out instance distributions, not only a training-matched test set.

Bipartite GNN branching. A MILP can be represented as a bipartite graph of variable and constraint nodes, with message passing producing a per-variable score used to select the branching variable, trained by behavioral cloning against strong-branching demonstrations. Reported results across generated MILP families (set covering, combinatorial auction, capacitated facility location, maximum independent set) are not uniformly positive: performance on maximum independent set is markedly worse and more variable than on the other families. This is controlled evidence that a single always-on learned branching policy has family-dependent failure modes — but it does not tell us whether the policy itself carries any usable signal about which of its own decisions are unreliable.

Integrated neural solver. A combined neural-branching and neural-diving system, evaluated end-to-end against a strong classical solver across several benchmark and application datasets, shows dataset- and time-horizon-dependent benefit: which learned component drives the improvement varies by setting. This is evidence that "does the learned component help" is not a single yes/no answer, which supports building a mechanism that can behave differently across decisions rather than assuming one policy configuration is globally best.

The gap. None of the above work asks whether a policy's own output distribution can be used, at inference time, to detect the decisions most likely to be unreliable. This proposal treats that as an open, testable question, not an assumed result.

2. Problem Statement

Learned branching policies for MILP can provide computational advantages on some instance distributions but may degrade substantially under distributional shift. Existing evaluations primarily compare learned and classical branching policies as fixed strategies, leaving open whether unreliable learned decisions can be identified at inference time. This project investigates whether the output distribution of a bipartite GNN branching policy provides a useful per-node signal of branching-decision reliability. Specifically, it tests whether low-confidence decisions are associated with poorer branching-decision quality, and whether selectively replacing such decisions with a classical branching rule improves solver robustness under in-distribution, size-shift, and structural-shift conditions, without eliminating the computational benefits of learned branching where it works.

Research questions

RQ1 (prediction). Does per-node GNN confidence correlate with branching-decision quality?

RQ2 (intervention). Does confidence-based fallback improve solver-level performance relative to always-GNN and always-classical branching?

RQ3 (attribution of the effect). Does confidence-based fallback provide any benefit beyond simply applying classical branching at the same average frequency, chosen at random? This is the matched-rate random-fallback comparison, and it is what separates "the signal is informative" from "occasionally using the classical rule helps regardless of which decisions it replaces."

These three questions are kept explicit and separate. RQ2 is only worth asking in full if RQ1 shows a usable correlation; RQ3 is what makes a positive RQ2 result mean something.

Terminology note: "confidence" and "harm"

Two terms in the original framing need tightening.

"Confidence" here refers initially to the concentration of the model's output distribution over candidate variables (e.g., softmax margin, entropy) — not to a calibrated probability that the chosen variable is optimal. A concentrated, "confident" distribution can still be confidently wrong. Whether concentration behaves like useful confidence (i.e., correlates with decision quality) is exactly what RQ1 tests, not something assumed by using the word.

"Harm" is avoided as the primary term until an intervention has actually been tested, since it implies causality that a correlational measurement (RQ1) cannot establish on its own. Before RQ2's intervention, the project uses decision quality or decision risk instead: e.g., "does confidence predict low-quality decisions," not "does confidence predict harm." Causal language is reserved for results that come from the actual fallback intervention (RQ2/RQ3), not from correlational analysis (RQ1).

Hypothesis

Low-confidence branching decisions are disproportionately associated with lower decision quality (RQ1), and thresholding on confidence to trigger classical fallback recovers a meaningful part of the degradation seen on hard families and shifted distributions, while preserving most of the speedup seen on easy, in-distribution cases (RQ2), beyond what matched-rate random fallback achieves alone (RQ3).

What this proposal does not claim. It does not claim novelty for GNN branching, bipartite MILP encodings, confidence estimation methods, or fallback/gating mechanisms as individual techniques — each has prior art requiring separate review. The claimed contribution, if the hypotheses hold, is the controlled empirical answer to RQ1–RQ3 in this specific setting, under matched-compute, per-family, per-shift evaluation.

3. A Cleaner Causal Hierarchy: Avoiding Bad Attribution

A branching decision does not have a clean, isolated effect on final solver outcomes — later decisions, primal heuristics, cuts, presolve, and propagation all interact with it. Treating "resulting subtree size" as directly attributable to a single decision risks confounded conclusions. To avoid this, the project separates two levels of measurement rather than jumping from confidence straight to end-to-end solver metrics:

Confidence  →  Decision quality  →  Search behavior  →  Solver performance


Decision-level proxies (used for RQ1, measured locally at the node): agreement with the strong-branching choice, the strong-branching score/ranking of the variable the GNN chose, immediate bound improvement at that node, and "regret" relative to the best available candidate. These are the primary evidence for RQ1 because they are not confounded by everything that happens later in the tree.

Solver-level outcomes (used for RQ2/RQ3, measured over a full solve): primal-dual integral, time to target gap, nodes explored, instances solved within a limit. These remain the primary evidence for whether the intervention (gating) helps, since that is a claim about the whole solve, not about one decision.

Subtree size attributable to a single node may still be reported as a secondary diagnostic, but it is not used as the main evidence for RQ1, precisely because it cannot cleanly isolate one decision's contribution.

4. Proposed Method

Base policy. Train a bipartite GCNN branching policy by behavioral cloning against strong-branching demonstrations, following the established architecture (variable and constraint node sets, message passing, softmax policy head over eligible fractional variables). One architecture only for the core project.

Confidence signals — primary set. Two statistics computed directly from the existing softmax output, requiring no additional model training and expected to have low inference cost because they are derived from a distribution the policy already produces: softmax margin (top-1 minus top-2 probability) and entropy of the candidate distribution. "Expected to be low-cost" is not the same as "negligible" — Section 8 requires this overhead to be measured, not assumed.

Confidence signals — optional extension, not core. Ensemble disagreement or Monte Carlo dropout are explicitly deferred to an optional extension, attempted only if the MVP (Section 6) succeeds. Including them in the core project risks turning it into a comparison of uncertainty-estimation techniques, which is a different research question than the one this proposal asks.

Decision-level calibration check (RQ1). Before building any gate, test directly whether the primary confidence signals correlate with the decision-level proxies in Section 3, on a logged sample of branching nodes across all four families. This is a stop/go step: if neither signal shows a usable correlation, RQ1 is answered negatively and the project reports that result rather than building a gate on a signal shown not to work.

Gate (RQ2/RQ3). If Stage 4 is positive: gate(confidence, τ) — below threshold τ, use a classical rule (solver-default or reliability/pseudo-cost branching); above τ, use the GNN's choice. τ is selected on a validation split only, never on final test outcomes.

Matched-rate random fallback (RQ3). A control that replaces the same average fraction of decisions with the classical rule, chosen uniformly at random rather than by confidence. This isolates whether the confidence signal itself carries information, independent of the general effect of occasionally deferring to the classical rule.

5. Datasets and Instance Generation

Families: set covering, combinatorial auction, capacitated facility location, and maximum independent set, reusing established generated-instance families for comparability with prior work; maximum independent set is included specifically because it is the documented failure case.

Splits, reported separately, never pooled: in-distribution test; held-out size shift (larger instances than training); held-out structural shift (changed density or coefficient distribution within the same family).

Verification step: exact instance-generation parameters, seeds, and train/validation/test counts must be checked against the original sources before generation begins; prior extracted summaries used to scope this proposal are secondary notes, not a substitute for primary method sections.

Labels: strong-branching demonstrations for the base policy; the decision-level proxies in Section 3 reuse strong-branching evaluations already available from that process, so no separate expert is needed to test RQ1.

6. Minimum Viable Experiment (MVP) — Scope Before Extensions

Given the number of moving parts (policy training, four families, solver instrumentation, node-level logging, gate design, two OOD shift types, multiple baselines, statistical comparison, failure analysis), the project defines an explicit MVP before attempting anything beyond it:

One GNN architecture (the established bipartite GCNN).

Two confidence signals only: softmax margin and entropy.

Four families, three splits each (in-distribution, size shift, structural shift).

Four conditions compared: always-GNN, always-classical, confidence-gated, matched-rate random fallback.

RQ1 answered via decision-level proxies; RQ2/RQ3 answered via solver-level outcomes.

Ensemble methods, Monte Carlo dropout, additional architectures, and any secondary refinement are explicitly out of scope until the MVP produces a result (positive or negative) for RQ1–RQ3.

7. Baselines

Baseline

Question it isolates

Default classical brancher

Does the gated system beat a strong practical solver configuration?

Reliability / pseudo-cost branching

Does the GNN (gated or not) improve on cheap, solver-native rules?

Strong branching (as inference-time policy, cost permitting)

How close is the learned/gated policy to the costly expert, and is the expert viable at inference time?

Always-GNN (ungated)

What does gating add or cost relative to the base policy alone? — directly answers RQ2

Always-classical

Is there any GNN benefit at all in-distribution?

Matched-rate random fallback

Does gating on confidence outperform switching to classical branching at the same average frequency but at random nodes? — directly answers RQ3

Tuned classical solver

Does learning add value beyond a well-tuned classical baseline, under matched compute and with tuning data kept separate from the test set?

8. Evaluation Metrics

For RQ1 (decision-level, not confounded by later search): agreement with strong branching, strong-branching rank/score of the chosen variable, immediate bound improvement, regret relative to the best candidate; correlation/calibration statistics (e.g., AUROC of confidence for predicting a low-quality decision) between these and the confidence signals.

For RQ2/RQ3 (solver-level, primary evidence for whether the intervention helps):

Primal-dual integral (primary metric — rewards early gap closure).

Time to target gap; instances solved within a fixed time limit.

Nodes explored, reported only as a diagnostic, never a substitute for wall-clock time.

Total inference overhead of the confidence computation and the gate check itself, measured and reported explicitly — the research questions do not assume this is negligible; the claim is instead framed as keeping the additional overhead small relative to total solve time, and the experiment determines whether that holds, rather than assuming it up front.

Per-family and per-split breakdown (in-distribution, size shift, structural shift) — never a single pooled number.

9. Stages

Stage 1 — Reproduction and instrumentation.
Reproduce the bipartite GCNN branching policy on the four families. Successful reproduction of the base brancher — matching, within a reasonable tolerance, the established results on at least the in-distribution splits — is an explicit prerequisite milestone before proceeding to Stage 2. If reproduction stalls, that is itself the finding to report and scope down from, rather than an assumed-solved step on the way to the confidence work. Once reproduction is confirmed, instrument branch-and-bound to log, at every decision, the full output distribution, the decision-level proxies (strong-branching agreement, rank, bound improvement, regret), and — on a sampled subset, for cost reasons, and where feasible — downstream search statistics for secondary analysis. These downstream statistics are not treated as clean node-level labels, since Section 3 notes they are confounded by everything that happens later in the tree; they are diagnostic only. This produces the dataset for Stage 2; no gate is built yet.

Stage 2 — RQ1: calibration study (stop/go).
Test whether the two primary confidence signals (margin, entropy) correlate with the decision-level proxies from Stage 1, per family and per split. If neither shows a usable correlation on any family, RQ1 is answered negatively; the project reports this as a legitimate result — "the proposed confidence signal does not provide sufficient predictive information for selective fallback in this setting" — rather than proceeding to build and tune a gate on a signal already shown not to work.

Stage 3 — Gate design and threshold selection (only if Stage 2 is positive).
Implement the gate, select τ on a validation split only, and fix all hyperparameters before touching test splits.

Stage 4 — RQ2/RQ3: controlled comparison.
Run the full baseline set (Section 7) across all three splits, with fixed compute budgets and inference overhead measured and reported in every timing result. RQ2 is answered by gated-GNN vs. always-GNN / always-classical; RQ3 is answered by gated-GNN vs. matched-rate random fallback.

Stage 5 — Ablation and failure analysis.
Isolate the gate's contribution from the base policy's, focusing specifically on maximum independent set and the two shift conditions, since these are the cases the gate is meant to address. Characterize confidently-wrong cases explicitly — decisions where the model was concentrated (high confidence) but the decision-level proxies indicate low quality — since a confidence gate cannot catch these by construction, and their frequency bounds how much a confidence-only gate can ever help.

Stage 6 — Write-up and scope statement.
Report per-family, per-split results without pooling. State explicitly which of RQ1, RQ2, and RQ3 were resolved and in which direction. Note that success does not require the gated GNN to beat classical branching everywhere — a result where gating reduces degradation under shift while preserving most in-distribution gains, and clears the random-fallback bar, is a complete and sufficient answer to the research question. Note the project's scope is branching (dual-bound side) only, not diving or primal heuristics.

10. Risks and Limitations to State Up Front

A positive RQ1 correlation may be specific to this GCNN architecture and training procedure, not a general property of learned branching policies.

Confidently-wrong decisions are a structural blind spot of any confidence-based gate (Section 9, Stage 5); the project should report how often this occurs rather than treating the gate as a complete solution.

Solver-level attribution of a single decision's effect is inherently confounded by everything that happens later in the tree (Section 3); this is why decision-level proxies, not subtree size, carry the main evidential weight for RQ1.

Reproducibility requires exact reporting of instance generation, seeds, solver parameters, splits, time limits, and hardware; hyperparameters (including τ) must never be selected using final test outcomes.

This is a substantial project — solver instrumentation and controlled experimental setup are likely harder than the GNN modeling itself. The MVP in Section 6 exists specifically to keep the project feasible before any extension is attempted.
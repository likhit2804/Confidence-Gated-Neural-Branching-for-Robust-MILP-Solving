# Docker and Linux MVP Runbook

## Purpose

This document records why Docker was introduced, what the Linux environment provides, how the MVP data and baseline experiment were run, and how to reproduce the work.

The first milestone is deliberately small: reproduce the existing bipartite GCNN on set-covering instances before adding confidence signals or a fallback gate.

## Why Docker and Linux

The project depends on native scientific software: Ecole, SCIP/PySCIPOpt, PyTorch, PyTorch Geometric, NumPy, and SciPy. Their behavior depends on Python, operating-system, compiler, and native-library versions. Docker gives the experiment a fixed Linux userspace and a repeatable dependency boundary while keeping source files and outputs on the Windows host.

The upstream `learn2branch-ecole` workflow is designed around Linux/Conda-style paths and native solver components. Running it inside Docker avoids changing the research code to accommodate Windows-specific native-library and process behavior. Docker Desktop supplies the Linux container runtime; the repository is mounted into `/workspace`.

## Repository layout

```text
Confidence-Gated-Neural-Branching-for-Robust-MILP-Solving/
├── Dockerfile
├── MVP_PLAN.md
├── first_problem_statement.md
├── repos/
│   ├── learn2branch/       # instance generation/reference code
│   ├── learn2branch-ecole/ # GCNN training/evaluation
│   └── branch2learn/       # supporting reference code
├── implementation/
│   ├── scripts/             # project conversion utilities
│   ├── data/                # generated outputs
│   └── logs/                # timestamped logs
└── research papers/
```

## Building and checking Docker

From the project root:

```powershell
docker build -t confidence-branching .
docker run --rm confidence-branching python --version
```

Interactive inspection:

```powershell
docker run --rm -it `
  -v "${PWD}:/workspace" `
  -w /workspace confidence-branching bash
```

The image is CPU-oriented for the initial MVP. GPU support is not required.


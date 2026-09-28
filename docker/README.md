# Docker Environment

This Docker setup provides a Linux environment for the Ecole-based MILP branching project while keeping Windows as the host system.

## Prerequisites

- Docker Desktop for Windows
- Linux containers enabled
- WSL2 backend recommended

## Build

From the repository root:

```powershell
docker build -t confidence-branching .
```

## Run the confidence utility tests

```powershell
docker run --rm confidence-branching
```

## Open an interactive shell with the repository mounted

```powershell
docker run --rm -it -v "${PWD}:/workspace" confidence-branching bash
```

Inside the container:

```bash
python -m pytest implementation/tests -vv
```

The initial image is CPU-only. GPU support can be added later after the CPU reproduction works.

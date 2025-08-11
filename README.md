# Minesweeper EfficientZero

This project provides a minimal framework for training a reinforcement learning agent based on EfficientZero to play Minesweeper. The environment supports reveal, flag, and chord actions.

## Setup

This repository uses the `uv` environment manager. Install uv and create the environment:

```bash
uv pip install -r requirements.txt
```

## Training

Run a short training session:

```bash
uv python scripts/train.py --config configs/small.yaml
```

The training loop now performs a small Monte Carlo Tree Search at each step
using the network's `initial_inference` and `recurrent_inference` functions to
produce a policy target for optimisation.

## Evaluation

```bash
uv python scripts/eval.py --checkpoint checkpoints/latest.pt
```

## UI

```bash
uv python scripts/ui.py
```

## Hardware

Designed for CUDA-enabled GPUs. Ensure `torch.cuda.is_available()` returns True.

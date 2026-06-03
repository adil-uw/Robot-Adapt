# Phase 4 — Multi-Terrain Environments

This document describes the two terrains used in RobotAdapt and how to use them in training and evaluation.

## Overview

| Terrain | Gymnasium ID | Description |
|---------|--------------|-------------|
| **Terrain A — Normal** | `Ant-v5` | Standard Gymnasium MuJoCo Ant locomotion task |
| **Terrain B — Slippery** | `AntSlippery-v5` | Same Ant task with **reduced floor friction** (icy conditions) |

Both terrains share the **same observation space, action space, reward function, and episode length**. Only contact friction on the floor geom differs. That makes transfer-drop experiments fair: performance changes reflect terrain adaptation, not a different task definition.

## Friction parameters

MuJoCo geom friction is `(sliding, torsional, rolling)`.

| Geom | Normal (`Ant-v5`) | Slippery (`AntSlippery-v5`) |
|------|-------------------|-----------------------------|
| `floor` | `(1.0, 0.5, 0.5)` | `(0.05, 0.005, 0.0001)` |

Implementation: `environments/friction_wrapper.py` wraps `Ant-v5` and modifies the `floor` geom only. Ant body geoms are unchanged.

### Important MuJoCo detail (why we set floor priority)

MuJoCo combines contact friction between two geoms using a **priority / maximum** rule. Since Ant's feet geoms have friction `(1.0, 0.5, 0.5)`, simply lowering the floor friction is not enough: MuJoCo would otherwise take the element-wise maximum and the contact would behave as if friction were still high.

Therefore `AntSlippery-v5` sets the **floor geom priority** higher than the feet so the floor's low friction dominates the contact.

## Usage

### Python factory (recommended)

```python
from environments import make_terrain_env

normal = make_terrain_env("normal")
slippery = make_terrain_env("slippery")
```

### Gymnasium IDs

```python
import gymnasium as gym
import environments  # registers AntSlippery-v5

normal = gym.make("Ant-v5")
slippery = gym.make("AntSlippery-v5")
```

### Training scripts

Pass the env id to existing trainers:

```powershell
# Normal terrain (unchanged)
python -m training.train_sac_baseline --env Ant-v5 ...

# Slippery terrain
python -m training.train_sac_baseline --env AntSlippery-v5 ...
```

### Evaluation and transfer

```powershell
# Evaluate on slippery terrain
python -m evaluation.evaluate --env AntSlippery-v5 --auto-discover --discover-algo SAC ...

# Transfer drop: normal -> slippery
python -m evaluation.transfer `
  --source-env Ant-v5 `
  --target-env AntSlippery-v5 `
  --sac training/models/phase3/sac/sac_normal_1000000_steps
```

## Validation

Run automated checks (spaces, friction, rollouts, eval framework compatibility):

```powershell
python -m environments.validate_terrains
```

## Files

| File | Purpose |
|------|---------|
| `friction_wrapper.py` | MuJoCo friction wrapper |
| `terrains.py` | Terrain factory, constants, Gymnasium registration |
| `validate_terrains.py` | Environment validation + compatibility tests |
| `TERRAIN.md` | This document |

## Design rationale

The professor requested transfer from **normal terrain to slippery / icy terrain**. Modifying floor friction is a standard, reproducible way to simulate ice without changing the Ant model or task structure. Keeping the same obs/action spaces ensures Phase 2 evaluation metrics and Phase 9 transfer-drop scripts work without modification.

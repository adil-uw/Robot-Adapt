# Robot-Adapt: Policy Distillation for Terrain Adaptation

A reinforcement learning study on the MuJoCo **Ant** robot that asks a simple
question: *what is the best way to get a quadruped to walk on multiple terrains
(normal ground and slippery ice)?*

We compare three strategies:

1. **Specialist teachers** — strong policies each trained on a single terrain.
2. **Student from scratch** — one policy trained directly on *mixed* terrain with RL.
3. **Distilled student** — one policy trained to *imitate* the specialist teachers.

The headline result is that the **distilled student** transfers across terrains
more robustly (higher return and forward distance, lower fall rate) than a
student trained from scratch on mixed terrain.

---

## Project structure

```
Robot-Adapt/
├── environments/                # Terrain definitions (normal + slippery)
│   ├── friction_wrapper.py      # MuJoCo wrapper that edits floor friction
│   ├── terrains.py              # Registers Ant-v5 (normal) + AntSlippery-v5 (icy)
│   ├── validate_terrains.py     # Sanity checks for the terrains
│   └── TERRAIN.md               # Detailed terrain documentation
├── training/                    # All training entry points
│   ├── train_ppo_baseline.py    # PPO baseline (normal terrain) with checkpoints
│   ├── train_sac_baseline.py    # SAC baseline / specialist teacher training
│   ├── train_student_scratch.py # Phase 6: student trained from scratch on mixed terrain
│   ├── run_distillation.py      # Phase 7: distill teacher(s) into a student
│   ├── watch_trained_agent.py   # Live MuJoCo viewer for a trained policy
│   └── distillation/            # Distillation building blocks
│       ├── student.py           # Student policy network (MLP -> Gaussian)
│       ├── teachers.py          # Teacher interfaces (SB3 + mock)
│       ├── data.py              # Collects states + teacher action distributions
│       ├── losses.py            # Gaussian KL-divergence loss
│       └── distill.py           # DistillationTrainer (+ TensorBoard logging)
├── evaluation/                  # Phase 2 evaluation framework
│   ├── evaluate.py              # Per-terrain metric tables + plots + CSV
│   ├── transfer.py              # Transfer-drop (normal -> slippery) report
│   ├── metrics.py               # Episode rollout + metric aggregation
│   ├── reporting.py             # Tables, CSV export, plots, best-checkpoint select
│   ├── plot_training_curves.py  # Static PNGs from TensorBoard logs
│   └── results/                 # Generated tables, plots, and curves
├── requirements.txt
└── README.md
```

> `models/`, `logs/`, `checkpoints/`, and the virtual environment are git-ignored.
> Trained checkpoints are large and are shared out-of-band rather than committed.

---

## Setup

Requires Python 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies (see `requirements.txt`):

- `gymnasium[mujoco]` — RL environments + MuJoCo physics
- `stable-baselines3` — PPO / SAC implementations
- `torch` — student policy network + distillation
- `numpy`, `matplotlib` — metrics and plots

---

## The two terrains

| Terrain | Gymnasium ID | Floor friction `(slide, torsion, roll)` |
|---|---|---|
| Normal | `Ant-v5` | `(1.0, 0.5, 0.5)` |
| Slippery (ice) | `AntSlippery-v5` | `(0.05, 0.005, 0.0001)` |

Both share the same observation space (27 numbers: body height, orientation,
joint angles, and all corresponding velocities), action space, reward, and
1000-step episode limit — only the floor friction changes, which keeps
transfer comparisons fair. See [`environments/TERRAIN.md`](environments/TERRAIN.md)
for the MuJoCo friction-priority detail.

---

## Usage

All commands are run from the repository root with the virtual environment
active (or by prefixing `./.venv/bin/python`).

### 1. Train baselines / teachers

```bash
# PPO baseline on normal terrain (saves checkpoints at milestones)
python -m training.train_ppo_baseline

# SAC specialist teacher on normal terrain
python -m training.train_sac_baseline --env Ant-v5

# SAC specialist teacher on slippery terrain
python -m training.train_sac_baseline --env AntSlippery-v5
```

### 2. Train the student from scratch (Phase 6)

Trains one policy on **mixed** terrain — each episode randomly starts on normal
or slippery ground, so the agent must learn a policy that copes with both.

```bash
python -m training.train_student_scratch \
    --algo SAC --milestones 100000 250000 500000 1000000 --seed 0
```

### 3. Distill teachers into a student (Phase 7)

The student learns by *imitating* the teachers (supervised KL-divergence on the
states the teachers visited) — no reward, no environment interaction during
learning.

```bash
python -m training.run_distillation \
    --teacher sac:training/models/phase5/teachers/normal/sac_normal_1000000_steps \
    --env Ant-v5 \
    --teacher sac:training/models/phase5/teachers/slippery/teacher_slippery_sac_1000000_steps \
    --env AntSlippery-v5 \
    --epochs 20 --steps-per-teacher 3000 \
    --out training/models/distilled_student.pt
```

Use `--teacher mock` for a quick smoke test without trained checkpoints.

### 4. Evaluate (Phase 2)

Evaluation is always done on **one terrain at a time**. `N = episodes × seeds`.

```bash
# Normal terrain
python -m evaluation.evaluate --env Ant-v5 \
    --episodes 20 --seeds 0 1 2 \
    --sac training/models/student_scratch_sac_1000000_steps

# Slippery terrain
python -m evaluation.evaluate --env AntSlippery-v5 \
    --episodes 20 --seeds 0 1 2 \
    --sac training/models/student_scratch_sac_1000000_steps

# Distilled student (.pt)
python -m evaluation.evaluate --env Ant-v5 \
    --episodes 20 --seeds 0 1 2 \
    --student training/models/distilled_student.pt

# Auto-discover every checkpoint in a folder + pick the best
python -m evaluation.evaluate --auto-discover \
    --models-dir training/models --select-best
```

Reported metrics: **Return** (total reward = forward progress + survival −
energy), **Length** (steps survived), **FwdDist** (meters walked), **Vel**,
**Energy** (control cost), **FallRate** (fraction of episodes that ended in a
fall).

### 5. Transfer drop (normal → slippery)

```bash
python -m evaluation.transfer \
    --source-env Ant-v5 --target-env AntSlippery-v5 \
    --episodes 20 --seeds 0 1 2 \
    --sac training/models/student_scratch_sac_1000000_steps
```

Swap `--sac` for `--student training/models/distilled_student.pt` to compare the
distilled student's transfer drop against the from-scratch student.

### 6. Watch a trained agent

Opens a live MuJoCo window. By default `watch_trained_agent.py` drops the agent
onto a low-friction floor to demonstrate slipping; raise `sliding_friction`
inside the script to watch normal walking.

```bash
python training/watch_trained_agent.py
```

### 7. Training curves (TensorBoard)

```bash
# Live dashboard
tensorboard --logdir training/logs

# Static PNGs for the report
python -m evaluation.plot_training_curves
```

---

## Results summary

Evaluated over 60 episodes (`--episodes 20 --seeds 0 1 2`) per terrain:

| Terrain | Policy | Return | FwdDist | FallRate |
|---|---|---|---|---|
| Normal | Distilled student | 2190.5 | 118.3 | 0.233 |
| Normal | From-scratch student | 1664.5 | 65.1 | 0.50 |
| Slippery | Distilled student | 2490.4 | 128.8 | 0.00 |
| Slippery | From-scratch student | 2413.3 | 92.1 | 0.00 |

The distilled student wins on both terrains — higher return and forward
distance, and a much lower fall rate on normal ground — supporting the core
claim that distilling from specialist teachers yields a more robust policy than
training from scratch on mixed terrain.

---

## Team

- **Lakshman** — evaluation framework (Phase 2), distillation pipeline (Phase 7),
  student-from-scratch training (Phase 6), transfer/metrics, reporting.
- **Adil** — SAC baseline (Phase 3/4) and terrain/friction environments.

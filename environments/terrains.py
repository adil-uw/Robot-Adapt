"""Phase 4 terrain environments: normal Ant-v5 and slippery (icy) variant.

Registered Gymnasium ids
------------------------
* ``Ant-v5``            — normal terrain (stock Gymnasium env)
* ``AntSlippery-v5``    — same Ant task with reduced floor friction

Use :func:`make_terrain_env` anywhere the project needs a terrain-aware factory
(training, evaluation, transfer tests).
"""

from __future__ import annotations

from typing import Literal

import gymnasium as gym
import mujoco

from environments.friction_wrapper import FrictionWrapper

TerrainId = Literal["normal", "slippery"]

# Stock Ant-v5 floor friction is [1.0, 0.5, 0.5] on geom "floor".
# Icy/slippery values follow common MuJoCo locomotion transfer setups.
SLIPPERY_FLOOR_FRICTION = (0.05, 0.005, 0.0001)

NORMAL_ENV_ID = "Ant-v5"
SLIPPERY_ENV_ID = "AntSlippery-v5"

TERRAIN_ENV_IDS: dict[TerrainId, str] = {
    "normal": NORMAL_ENV_ID,
    "slippery": SLIPPERY_ENV_ID,
}


def _make_slippery_ant(**kwargs) -> gym.Env:
    """Build Ant-v5 with icy floor friction."""
    base = gym.make(NORMAL_ENV_ID, **kwargs)
    sliding, torsional, rolling = SLIPPERY_FLOOR_FRICTION
    return FrictionWrapper(
        base,
        sliding_friction=sliding,
        torsional_friction=torsional,
        rolling_friction=rolling,
        # MuJoCo combines contact friction using geom priority / max rule.
        # Set floor priority higher than the feet so floor friction dominates.
        priority=1,
    )


def register_terrains() -> None:
    """Register custom terrain env ids with Gymnasium (safe to call repeatedly)."""
    if SLIPPERY_ENV_ID not in gym.envs.registry:
        gym.register(
            id=SLIPPERY_ENV_ID,
            entry_point="environments.terrains:_make_slippery_ant",
            max_episode_steps=1000,
            reward_threshold=6000.0,
        )


def make_terrain_env(
    terrain: TerrainId | str = "normal",
    *,
    render_mode: str | None = None,
    **kwargs,
) -> gym.Env:
    """Create an environment for the requested terrain.

    Parameters
    ----------
    terrain:
        ``"normal"`` → ``Ant-v5``; ``"slippery"`` → ``AntSlippery-v5``.
    render_mode:
        Optional Gymnasium render mode (e.g. ``"human"``).
    """
    register_terrains()
    terrain_key = terrain.lower()  # type: ignore[assignment]
    if terrain_key not in TERRAIN_ENV_IDS:
        raise ValueError(f"Unknown terrain {terrain!r}. Use 'normal' or 'slippery'.")

    env_id = TERRAIN_ENV_IDS[terrain_key]  # type: ignore[index]
    make_kwargs = dict(kwargs)
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode
    return gym.make(env_id, **make_kwargs)


def get_floor_friction(env: gym.Env) -> dict[str, tuple[float, float, float]]:
    """Read floor geom friction from a normal or wrapped Ant env."""
    model = env.unwrapped.model
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("Floor geom not found.")
    values = tuple(float(x) for x in model.geom_friction[floor_id])
    return {"floor": values}


# Register on import so ``gym.make("AntSlippery-v5")`` works project-wide.
register_terrains()

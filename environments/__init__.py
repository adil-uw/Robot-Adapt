"""Multi-terrain environments for RobotAdapt (Phase 4)."""

from environments.friction_wrapper import FrictionWrapper
from environments.terrains import (
    NORMAL_ENV_ID,
    SLIPPERY_ENV_ID,
    SLIPPERY_FLOOR_FRICTION,
    TERRAIN_ENV_IDS,
    make_terrain_env,
    register_terrains,
)

__all__ = [
    "FrictionWrapper",
    "NORMAL_ENV_ID",
    "SLIPPERY_ENV_ID",
    "SLIPPERY_FLOOR_FRICTION",
    "TERRAIN_ENV_IDS",
    "make_terrain_env",
    "register_terrains",
]

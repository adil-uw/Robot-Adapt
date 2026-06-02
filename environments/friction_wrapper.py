"""MuJoCo friction wrapper for terrain modification (Phase 4).

Gymnasium MuJoCo envs expose a shared ``model`` object. This wrapper scales or
sets sliding / torsional / rolling friction on selected geoms (default: floor).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import gymnasium as gym
import mujoco
import numpy as np

# MuJoCo geom friction is (sliding, torsional, rolling).
DEFAULT_FLOOR_GEOM_NAMES = ("floor",)


class FrictionWrapper(gym.Wrapper):
    """Modify contact friction on selected MuJoCo geoms."""

    def __init__(
        self,
        env: gym.Env,
        *,
        geom_names: Iterable[str] = DEFAULT_FLOOR_GEOM_NAMES,
        sliding_friction: float | None = None,
        torsional_friction: float | None = None,
        rolling_friction: float | None = None,
        friction_scale: float | None = None,
    ):
        super().__init__(env)
        self.geom_names = tuple(geom_names)
        self.sliding_friction = sliding_friction
        self.torsional_friction = torsional_friction
        self.rolling_friction = rolling_friction
        self.friction_scale = friction_scale

        if friction_scale is None and sliding_friction is None:
            raise ValueError("Set friction_scale or sliding_friction (or both).")

        self._model = env.unwrapped.model
        self._geom_ids = self._resolve_geom_ids()
        self._original_friction = {
            geom_id: self._model.geom_friction[geom_id].copy()
            for geom_id in self._geom_ids
        }
        self._apply_friction()

    def _resolve_geom_ids(self) -> list[int]:
        geom_ids: list[int] = []
        for name in self.geom_names:
            geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id < 0:
                raise ValueError(f"Geom '{name}' not found in MuJoCo model.")
            geom_ids.append(geom_id)
        return geom_ids

    def _apply_friction(self) -> None:
        for geom_id in self._geom_ids:
            original = self._original_friction[geom_id]
            sliding = (
                float(original[0] * self.friction_scale)
                if self.friction_scale is not None and self.sliding_friction is None
                else float(self.sliding_friction if self.sliding_friction is not None else original[0])
            )
            torsional = (
                float(original[1] * self.friction_scale)
                if self.friction_scale is not None and self.torsional_friction is None
                else float(
                    self.torsional_friction if self.torsional_friction is not None else original[1]
                )
            )
            rolling = (
                float(original[2] * self.friction_scale)
                if self.friction_scale is not None and self.rolling_friction is None
                else float(
                    self.rolling_friction if self.rolling_friction is not None else original[2]
                )
            )
            self._model.geom_friction[geom_id] = np.array(
                [sliding, torsional, rolling], dtype=np.float64
            )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._apply_friction()
        return obs, info

    def get_floor_friction(self) -> dict[str, Sequence[float]]:
        """Return current friction values for wrapped geoms (for validation)."""
        out: dict[str, Sequence[float]] = {}
        for geom_id in self._geom_ids:
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            out[name or str(geom_id)] = tuple(float(x) for x in self._model.geom_friction[geom_id])
        return out

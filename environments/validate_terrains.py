"""Phase 4: validate normal vs slippery terrains and compatibility checks.

Run from project root::

    python -m environments.validate_terrains
"""

from __future__ import annotations

import gymnasium as gym

from environments.terrains import (
    NORMAL_ENV_ID,
    SLIPPERY_ENV_ID,
    SLIPPERY_FLOOR_FRICTION,
    get_floor_friction,
    make_terrain_env,
    register_terrains,
)
from evaluation.metrics import evaluate_policy
from evaluation.policies import random_policy


def _assert_spaces_match(normal: gym.Env, slippery: gym.Env) -> None:
    assert normal.observation_space == slippery.observation_space, "observation_space mismatch"
    assert normal.action_space == slippery.action_space, "action_space mismatch"
    print("[ok] observation and action spaces match")


def _assert_friction_differs(normal: gym.Env, slippery: gym.Env) -> None:
    normal_floor = get_floor_friction(normal)["floor"]
    slippery_floor = get_floor_friction(slippery)["floor"]
    print(f"  normal floor friction:   {normal_floor}")
    print(f"  slippery floor friction: {slippery_floor}")
    assert normal_floor[0] > slippery_floor[0], "sliding friction should be lower on slippery terrain"
    assert slippery_floor == SLIPPERY_FLOOR_FRICTION, "slippery floor friction mismatch"
    print("[ok] floor friction differs (slippery < normal)")


def _simple_rollout(env: gym.Env, steps: int = 200, seed: int = 0) -> None:
    policy = random_policy(env)
    obs, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = policy(obs)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"[ok] {steps}-step random rollout completed")


def _compatibility_with_eval_framework() -> None:
    """Ensure evaluation.metrics works with both env ids."""
    register_terrains()

    def normal_factory() -> gym.Env:
        return gym.make(NORMAL_ENV_ID)

    def slippery_factory() -> gym.Env:
        return gym.make(SLIPPERY_ENV_ID)

    for name, factory in (("normal", normal_factory), ("slippery", slippery_factory)):
        probe = factory()
        rand = random_policy(probe)
        probe.close()
        result = evaluate_policy(factory, rand, name, num_episodes=2, seeds=(0,))
        assert result.num_episodes == 2
        print(
            f"[ok] evaluation framework on {name}: "
            f"return={result.avg_return:.2f}, fall_rate={result.fall_rate:.2f}"
        )


def main() -> None:
    print("Phase 4 terrain validation\n" + "=" * 40)
    register_terrains()

    print("\n1) Environment creation")
    normal = make_terrain_env("normal")
    slippery = make_terrain_env("slippery")
    print(f"  normal env id:   {NORMAL_ENV_ID}")
    print(f"  slippery env id: {SLIPPERY_ENV_ID}")

    print("\n2) Space compatibility")
    _assert_spaces_match(normal, slippery)

    print("\n3) Friction modification")
    _assert_friction_differs(normal, slippery)

    print("\n4) Rollout validation")
    print("  normal:")
    _simple_rollout(normal, steps=200)
    print("  slippery:")
    _simple_rollout(slippery, steps=200)

    print("\n5) Evaluation framework compatibility")
    _compatibility_with_eval_framework()

    normal.close()
    slippery.close()
    print("\nAll Phase 4 terrain checks passed.")


if __name__ == "__main__":
    main()

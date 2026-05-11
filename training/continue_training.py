import gymnasium as gym
from stable_baselines3 import PPO


# --------------------------------------------------
# Create the same Ant environment used before
# --------------------------------------------------
env = gym.make("Ant-v5")


# --------------------------------------------------
# Load the already good 10k PPO model
# --------------------------------------------------
# This continues from the model that performed well.
model = PPO.load(
    "models/ppo_ant",
    env=env
)


# --------------------------------------------------
# Continue training for more timesteps
# --------------------------------------------------
# reset_num_timesteps=False keeps TensorBoard timeline continuous.
model.learn(
    total_timesteps=100_000,
    reset_num_timesteps=False
)


# --------------------------------------------------
# Save the continued model separately
# --------------------------------------------------
model.save("models/ppo_ant_10k_plus_100k")


# --------------------------------------------------
# Close environment
# --------------------------------------------------
env.close()

print("Continued training finished.")
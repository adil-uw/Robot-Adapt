import os
import gymnasium as gym
from stable_baselines3 import PPO


# -----------------------------
# 1. Create folders if missing
# -----------------------------
# These folders will store trained models and training logs.
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


# -----------------------------
# 2. Create the Ant environment
# -----------------------------
# render_mode is NOT used during training because rendering slows training.
# We only render later when we want to watch the trained robot.
env = gym.make("Ant-v5")


# -----------------------------
# 3. Create PPO model
# -----------------------------
# MlpPolicy means PPO will use a normal neural network.
# It takes Ant observations as input and outputs Ant actions.
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="logs/"
)


# -----------------------------
# 4. Train the model
# -----------------------------
# total_timesteps means how many environment steps PPO will train for.
# 10,000 is small, good for first test.
# Later we may increase this to 100,000 or more.
model.learn(total_timesteps=100_000)


# -----------------------------
# 5. Save the trained model
# -----------------------------
# This creates models/ppo_ant.zip
model.save("models/ppo_ant_100k_steps")


# -----------------------------
# 6. Close the environment
# -----------------------------
env.close()

print("Training finished. Model saved at models/ppo_ant.zip")
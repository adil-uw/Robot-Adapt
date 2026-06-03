import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC


# Create Ant environment with rendering enabled.
# render_mode="human" opens the MuJoCo window so we can watch the robot.
env = gym.make("Ant-v5", render_mode="human")


# Load the PPO model we trained earlier.
# This reads models/ppo_ant.zip from the models folder.
model = PPO.load("models/ppo_ant_100k_steps")
#model = PPO.load("training/models/phase3/ppo/ppo_normal_1000000_steps")
#model = SAC.load("training/models/phase3/sac/sac_normal_1000000_steps")
# Reset the environment before starting.
# obs contains the first observation/state of the Ant.
obs, info = env.reset()


# Run forever so we can keep watching the trained robot.
for step in range(3000):
#while True:
    # Ask the trained model what action to take for the current observation.
    # deterministic=True means use the best learned action, not random exploration.
    action, _states = model.predict(obs, deterministic=True)

    # Apply the action to the Ant simulation.
    obs, reward, terminated, truncated, info = env.step(action)

    # If the episode ends, restart the Ant.
    if terminated or truncated:
        obs, info = env.reset() 

env.close()
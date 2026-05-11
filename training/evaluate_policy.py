import gymnasium as gym
from stable_baselines3 import PPO


# --------------------------------------------------
# Evaluate RANDOM policy
# --------------------------------------------------
def evaluate_random_policy(env_name, num_episodes=5):

    # Create environment
    env = gym.make(env_name)

    # Store rewards from all episodes
    episode_rewards = []

    for episode in range(num_episodes):

        # Reset environment
        obs, info = env.reset()

        # Track total reward
        total_reward = 0

        while True:

            # Random action
            action = env.action_space.sample()

            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)

            # Accumulate reward
            total_reward += reward

            # Stop episode if done
            if terminated or truncated:
                break

        # Save episode reward
        episode_rewards.append(total_reward)

    # Close environment
    env.close()

    # Return average reward
    return sum(episode_rewards) / len(episode_rewards)


# --------------------------------------------------
# Evaluate TRAINED PPO model
# --------------------------------------------------
def evaluate_trained_policy(env_name, model_path, num_episodes=5):

    # Create environment
    env = gym.make(env_name)

    # Load PPO model
    model = PPO.load(model_path)

    # Store rewards
    episode_rewards = []

    for episode in range(num_episodes):

        # Reset environment
        obs, info = env.reset()

        total_reward = 0

        while True:

            # PPO predicts action
            action, _states = model.predict(
                obs,
                deterministic=True
            )

            # Apply action
            obs, reward, terminated, truncated, info = env.step(action)

            # Add reward
            total_reward += reward

            # Stop if episode ends
            if terminated or truncated:
                break

        # Save reward
        episode_rewards.append(total_reward)

    # Close environment
    env.close()

    # Return average reward
    return sum(episode_rewards) / len(episode_rewards)


# --------------------------------------------------
# Main Evaluation Section
# --------------------------------------------------

env_name = "Ant-v5"


# Evaluate random policy
random_reward = evaluate_random_policy(
    env_name,
    num_episodes=10
)


# Evaluate PPO 10k model
ppo_10k_reward = evaluate_trained_policy(
    env_name,
    "models/ppo_ant",
    num_episodes=10
)


# Evaluate PPO 100k model
ppo_100k_reward = evaluate_trained_policy(
    env_name,
    "models/ppo_ant_100k_steps",
    num_episodes=10
)

# Evaluate PPO model that continued training from the good 10k checkpoint
ppo_continued_reward = evaluate_trained_policy(
    env_name,
    "models/ppo_ant_10k_plus_100k",
    num_episodes=10
)
# --------------------------------------------------
# Print final comparison results
# --------------------------------------------------
print("Evaluation Results")
print("------------------")

print(f"Random Policy Average Reward: {random_reward}")

print(f"PPO 10k Average Reward: {ppo_10k_reward}")

print(f"PPO 100k Average Reward: {ppo_100k_reward}")

print(f"PPO 10k + 100k Continued Average Reward: {ppo_continued_reward}")

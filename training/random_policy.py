import gymnasium as gym


# Create the Ant-v5 MuJoCo environment with a visible viewer window.
# render_mode="human" opens the simulator so we can SEE the robot move.
# This script does NOT train anything — it only runs random actions
# to give us a baseline of how a completely untrained Ant performs.
env = gym.make("Ant-v5", render_mode="human")


# How many full episodes to run. An episode ends when the Ant falls over
# (terminated) or hits the time limit (truncated).
num_episodes = 3


for episode in range(num_episodes):
    # Reset the environment at the start of every episode.
    # obs  = the starting observation (joint angles, velocities, etc.)
    # info = extra debug info from Gymnasium (we don't use it here).
    obs, info = env.reset()

    # Track how well this episode went.
    total_reward = 0   # sum of rewards across the episode
    step_count = 0     # how many steps the Ant survived

    # Loop forever until the episode ends.
    while True:
        # Sample a completely random action from the action space.
        # No learning, no policy — just pure noise. This is our baseline.
        action = env.action_space.sample()

        # Apply the random action to the simulator and read the result.
        # reward     = how good this step was (forward progress, alive bonus, etc.)
        # terminated = True if the Ant fell / failed
        # truncated  = True if the env hit its max step limit
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Stop the episode the moment it ends.
        if terminated or truncated:
            break

    # Print a quick summary so we can compare against a trained agent later.
    print(f"Episode {episode + 1}")
    print(f"Total Reward: {total_reward}")
    print(f"Steps Survived: {step_count}")
    print("--------------------")


# Close the simulator window and free MuJoCo resources.
env.close()

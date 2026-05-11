import gymnasium as gym

env = gym.make("Ant-v5", render_mode="human")

obs, info = env.reset()

print("Observation Space:", env.observation_space)
print("Action Space:", env.action_space)

for _ in range(1000):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()
        print(obs)

env.close()
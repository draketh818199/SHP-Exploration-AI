import gymnasium as gym
import time
import PettingZooEnvironement
# for testing
from pettingzoo.butterfly import cooperative_pong_v5

# Create our training environment - a cart with a pole that needs balancing
#env = gym.make("GridMapEnv-v0", render_mode="human")

# create petting zoo environement
env = PettingZooEnvironement.env(render_mode="human")

# Reset environment to start a new episode
observation, reward, terminated, truncated, info = env.reset()

agent = "agent_0"
episode_over = False
total_reward = 0

while not episode_over:


    if (terminated[agent] or truncated[agent]):
        action = None

    else:
        action = env.action_space(agent).sample()  # Random action for now - real agents will be smarter!

    actions = {agent: action}
    print(action)
    observation, reward, terminated, truncated, info = env.step(actions)

    total_reward += reward[agent]
    episode_over = terminated[agent] or truncated[agent]
    time.sleep(.05)

print(f"Episode finished! Total reward: {total_reward}")
print(truncated, terminated)
env.close()
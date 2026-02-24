import gymnasium as gym
import time
import PettingZooEnvironement
# for testing
from pettingzoo.butterfly import cooperative_pong_v5


# create petting zoo environement (render_mode="human") to enable visuals
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
        # env.action_space is the list of all actions . sample gets a random action
        action = env.action_space(agent).sample()  # Random action for now - real agents will be smarter!

    # tells the environment that agent prefeomres action
    actions = {agent: action}
    # displays visual if enabled
    env.render()
    # gets output from enironment (most not used currently)
    observation, reward, terminated, truncated, info = env.step(actions)

    # sums reward
    total_reward += reward[agent]
    # Ends Session
    episode_over = terminated[agent] or truncated[agent]

#final output
print(f"Episode finished! Total reward: {total_reward}")
print(truncated, terminated)
# important to close environment
env.close()
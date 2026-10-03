from src.environment import create_taxi_env
from src.rewards import base_reward
from src.agent import Agent
import src.config as config

# Test the Agent's core functionality.

env = create_taxi_env()

agent = Agent(
    env=env,
    reward_function=base_reward,
    config=config
)

rewards = agent.train()
assert len(rewards) == config.NUM_EPISODES
print("Agent test passed.")
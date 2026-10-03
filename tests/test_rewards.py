# Tests to check that the reward functions return the expected values.

from src.environment import create_taxi_env
from src.rewards import (
    base_reward,
    sparse_reward,
    custom_reward_1,
    custom_reward_2,
)


def test_rewards():

    env = create_taxi_env()

    print("Base Reward:")
    print(base_reward(env, None, None, None, -1, False, 0))
    print(base_reward(env, None, None, None, 20, True, 0))
    print(base_reward(env, None, None, None, -10, False, 0))

    print("\nSparse Reward:")
    print(sparse_reward(env, None, None, None, -1, False, 0))
    print(sparse_reward(env, None, None, None, 20, True, 0))
    print(sparse_reward(env, None, None, None, -10, False, 0))

    print("\nCustom Reward 1:")
    print(custom_reward_1(env, None, None, None, -1, False, 0))
    print(custom_reward_1(env, None, None, None, 20, True, 10))
    print(custom_reward_1(env, None, None, None, -10, False, 0))

    env.close()


if __name__ == "__main__":
    test_rewards()
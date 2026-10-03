from experiments.experiment_runner import run_experiment

from src.rewards import (base_reward, sparse_reward, custom_reward_1, custom_reward_2, custom_reward_3,)

def main():
    run_experiment(base_reward, "Base Reward")
    print()
    run_experiment(sparse_reward, "Sparse Reward")
    print()
    run_experiment(custom_reward_1, "Custom Reward 1")
    print()
    run_experiment(custom_reward_2, "Custom Reward 2")
    print()
    run_experiment(custom_reward_3, "Custom Reward 3")

if __name__ == "__main__":
    main()
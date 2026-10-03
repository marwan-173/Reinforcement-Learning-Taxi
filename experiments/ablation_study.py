from experiments.experiment_runner import run_experiment
from src.rewards import base_reward, custom_reward_1


def run_ablation(reward_function, reward_name):

    print(f"\n========== {reward_name} : SARSA ==========")
    run_experiment(reward_function=reward_function, reward_name=f"{reward_name} SARSA", use_planning=False, use_kernel=False, save_policy=True,)

    print(f"\n========== {reward_name} : SARSA + Planning ==========")
    run_experiment(reward_function=reward_function, reward_name=f"{reward_name} Planning", use_planning=True, use_kernel=False, save_policy=True,)

    print(f"\n========== {reward_name} : SARSA + Kernel ==========")
    run_experiment(reward_function=reward_function, reward_name=f"{reward_name} Kernel", use_planning=False, use_kernel=True, save_policy=True,)

    print(f"\n========== {reward_name} : Full Algorithm ==========")
    run_experiment(reward_function=reward_function, reward_name=f"{reward_name} Full", use_planning=True, use_kernel=True, save_policy=True,)


def main():
    run_ablation(base_reward, "Base Reward")
    run_ablation(custom_reward_1, "Custom Reward 1")


if __name__ == "__main__":
    main()
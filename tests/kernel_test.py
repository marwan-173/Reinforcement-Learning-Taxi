from experiments.experiment_runner import run_experiment
from src.rewards import base_reward


def main():

    print("\n========== SARSA ==========")
    run_experiment(
        reward_function=base_reward,
        reward_name="Kernel_Test_SARSA",
        use_planning=False,
        use_kernel=False,
    )

    print("\n========== SARSA + Kernel ==========")
    run_experiment(
        reward_function=base_reward,
        reward_name="Kernel_Test_Kernel",
        use_planning=False,
        use_kernel=True,
    )


if __name__ == "__main__":
    main()
from experiments.experiment_runner import run_experiment
from src.rewards import base_reward, custom_reward_1
import src.config as config

def alpha_sweep(reward_function, reward_name):
    
    original_alpha = config.ALPHA
    alpha_values = [0.05, 0.1, 0.2, 0.4]
    for alpha in alpha_values:

        print(f"\n========== Alpha = {alpha} ==========")
        config.ALPHA = alpha
        run_experiment(reward_function=reward_function, reward_name=f"{reward_name} alpha {alpha}",
            use_planning=True, use_kernel=False,)

    config.ALPHA = original_alpha


def epsilon_sweep(reward_function, reward_name):

    original_epsilon = config.EPSILON

    epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3]

    for epsilon in epsilon_values:

        print(f"\n========== Epsilon = {epsilon} ==========")

        config.EPSILON = epsilon

        run_experiment(
            reward_function=reward_function,
            reward_name=f"{reward_name} epsilon {epsilon}",
            use_planning=True,
            use_kernel=False,
        )

    config.EPSILON = original_epsilon

    
def main():

    alpha_sweep(base_reward, "Base Reward")
    alpha_sweep(custom_reward_1, "Custom Reward 1")

    epsilon_sweep(base_reward, "Base Reward")
    epsilon_sweep(custom_reward_1, "Custom Reward 1")

if __name__ == "__main__":
    main()
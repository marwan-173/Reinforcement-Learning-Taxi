import os
import pandas as pd
import matplotlib.pyplot as plt

reward_files = {
    "Base": "base_reward.csv",
    "Sparse": "sparse_reward.csv",
    "Custom 1": "custom_reward_1.csv",
    "Custom 2": "custom_reward_2.csv",
    "Custom 3": "custom_reward_3.csv",
}

base_alpha = {
    "0.05": "base_reward_alpha_0.05.csv",
    "0.10": "base_reward_alpha_0.1.csv",
    "0.20": "base_reward_alpha_0.2.csv",
    "0.40": "base_reward_alpha_0.4.csv",
}

base_epsilon = {
    "0.01": "base_reward_epsilon_0.01.csv",
    "0.05": "base_reward_epsilon_0.05.csv",
    "0.10": "base_reward_epsilon_0.1.csv",
    "0.20": "base_reward_epsilon_0.2.csv",
    "0.30": "base_reward_epsilon_0.3.csv",
}

custom1_alpha = {
    "0.05": "custom_reward_1_alpha_0.05.csv",
    "0.10": "custom_reward_1_alpha_0.1.csv",
    "0.20": "custom_reward_1_alpha_0.2.csv",
    "0.40": "custom_reward_1_alpha_0.4.csv",
}

custom1_epsilon = {
    "0.01": "custom_reward_1_epsilon_0.01.csv",
    "0.05": "custom_reward_1_epsilon_0.05.csv",
    "0.10": "custom_reward_1_epsilon_0.1.csv",
    "0.20": "custom_reward_1_epsilon_0.2.csv",
    "0.30": "custom_reward_1_epsilon_0.3.csv",
}

base_ablation = {
    "SARSA": "base_reward_sarsa.csv",
    "Planning": "base_reward_planning.csv",
    "Kernel": "base_reward_kernel.csv",
    "Full": "base_reward_full.csv",
}

custom1_ablation = {
    "SARSA": "custom_reward_1_sarsa.csv",
    "Planning": "custom_reward_1_planning.csv",
    "Kernel": "custom_reward_1_kernel.csv",
    "Full": "custom_reward_1_full.csv",
}

def load_metric(csv_file, metric):

    dataframe = pd.read_csv(csv_file)
    grouped = dataframe.groupby("Episode")[metric]
    mean = grouped.mean()
    std = grouped.std()
    count = grouped.count()
    ci = 1.96 * std / (count ** 0.5)

    mean = mean.rolling(window=20, min_periods=1).mean()
    ci = ci.rolling(window=20, min_periods=1).mean()

    return mean.index, mean, ci


def plot_comparison(files, metric, ylabel, title, output_name):

    plt.figure(figsize=(10, 5))

    for label, filename in files.items():

        episodes, mean, ci = load_metric(f"results/csv/{filename}", metric)

        plt.plot(episodes, mean, label=label, linewidth=2)
        plt.fill_between( episodes, mean - ci, mean + ci, alpha=0.2)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"results/comparison_plots/{output_name}", dpi=300,)
    plt.close()

def main():

    #Reward Comparison
    os.makedirs("results/comparison_plots", exist_ok=True)
    plot_comparison(reward_files, "Reward", "Cumulative Reward",
        "Reward Comparison - Cumulative Reward","reward_comparison_reward.png",)

    plot_comparison( reward_files, "Episode Length", "Episode Length", 
        "Reward Comparison - Episode Length", "reward_comparison_episode_length.png",)

    #Base Alpha Sweep
    plot_comparison(base_alpha, "Reward", "Cumulative Reward",
        "Base Alpha Sweep - Cumulative Reward", "base_alpha_reward.png",)

    plot_comparison(base_alpha, "Episode Length", "Episode Length",
        "Base Alpha Sweep - Episode Length", "base_alpha_episode_length.png",)

    #Base Epsilon Sweep
    plot_comparison(base_epsilon, "Reward", "Cumulative Reward",
        "Base Epsilon Sweep - Cumulative Reward", "base_epsilon_reward.png",)

    plot_comparison(base_epsilon, "Episode Length", "Episode Length",
        "Base Epsilon Sweep - Episode Length", "base_epsilon_episode_length.png",)


    #Custom 1 Alpha Sweep
    plot_comparison(custom1_alpha, "Reward", "Cumulative Reward",
        "Custom 1 Alpha Sweep - Cumulative Reward", "custom1_alpha_reward.png",
    )

    plot_comparison(custom1_alpha, "Episode Length", "Episode Length",
        "Custom 1 Alpha Sweep - Episode Length", "custom1_alpha_episode_length.png",
    )


    #Custom 1 Epsilon Sweep
    plot_comparison(custom1_epsilon, "Reward", "Cumulative Reward",
        "Custom 1 Epsilon Sweep - Cumulative Reward", "custom1_epsilon_reward.png",)

    plot_comparison(custom1_epsilon, "Episode Length", "Episode Length",
        "Custom 1 Epsilon Sweep - Episode Length", "custom1_epsilon_episode_length.png",)


    #Base Ablation
    plot_comparison(base_ablation, "Reward", "Cumulative Reward",
        "Base Ablation - Cumulative Reward", "base_ablation_reward.png",)

    plot_comparison(base_ablation, "Episode Length", "Episode Length",
        "Base Ablation - Episode Length", "base_ablation_episode_length.png",)


    #Custom 1 Ablation
    plot_comparison(custom1_ablation, "Reward", "Cumulative Reward",
        "Custom 1 Ablation - Cumulative Reward", "custom1_ablation_reward.png",)

    plot_comparison(custom1_ablation, "Episode Length", "Episode Length",
        "Custom 1 Ablation - Episode Length", "custom1_ablation_episode_length.png",)

    print("All report plots generated successfully.")

if __name__ == "__main__":
    main()
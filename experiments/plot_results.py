import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_csv(csv_file):

    dataframe = pd.read_csv(csv_file)
    name = os.path.splitext(os.path.basename(csv_file))[0]

    plt.figure(figsize=(10, 5))

    # Single-seed experiment
    if dataframe["Seed"].nunique() == 1:

        smoothed_reward = (  dataframe["Reward"].rolling(window=20, min_periods=1).mean() )
        plt.plot(dataframe["Episode"], smoothed_reward, label="Moving Average (20)")

    # Multi-seed experiment
    else:

        grouped = dataframe.groupby("Episode")["Reward"]

        mean = grouped.mean()
        std = grouped.std()
        count = grouped.count()

        ci95 = 1.96 * std / np.sqrt(count)

        smoothed_mean = ( mean.rolling(window=20, min_periods=1).mean() )

        smoothed_upper = ( (mean + ci95).rolling(window=20, min_periods=1).mean() )
        smoothed_lower = ( (mean - ci95).rolling(window=20, min_periods=1).mean() )

        plt.plot(mean.index, smoothed_mean, label="Mean Reward")
        plt.fill_between(mean.index, smoothed_lower, smoothed_upper, alpha=0.2, label="95% Confidence Interval")

    plt.title(name.replace("_", " ").title())
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"results/plots/{name}.png")
    plt.close()


def main():

    os.makedirs("results/plots", exist_ok=True)
    for file in os.listdir("results/csv"):
        if file.endswith(".csv"):
            plot_csv(os.path.join("results/csv", file))


if __name__ == "__main__":
    main()
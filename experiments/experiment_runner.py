from src.environment import create_taxi_env
from src.agent import Agent
import src.config as config
import os
import pandas as pd
import numpy as np

def run_experiment(reward_function, reward_name, use_planning=True, use_kernel=False,
                     num_episodes=None, num_seeds=None, seed=None, save_policy=False):
    # Run one training experiment and return the collected episode metrics.

    if num_episodes is None:
        num_episodes = config.NUM_EPISODES

    if seed is None:
        seed = config.SEED

    if num_seeds is None:
        num_seeds = config.NUM_SEEDS

    old_num_episodes = config.NUM_EPISODES
    config.NUM_EPISODES = num_episodes
    results = []
    best_reward = float("-inf")
    best_q_table = None

    for current_seed in range(num_seeds):

        actual_seed = seed + current_seed
        np.random.seed(actual_seed)
        env = create_taxi_env(seed=actual_seed)
        env.action_space.seed(actual_seed)
        agent = Agent(env=env, reward_function=reward_function, config=config, use_planning=use_planning, use_kernel=use_kernel,)
        metrics = agent.train(description=f"{reward_name} | Seed {current_seed + 1}/{num_seeds}",
                            save_checkpoints=save_policy and use_planning and use_kernel,)

        for episode, metric in enumerate(metrics, start=1):

            results.append({
                "Seed": actual_seed,
                "Episode": episode,
                "Reward": metric.cumulative_reward,
                "Episode Length": metric.episode_length,
                "Success": metric.success,
                "Illegal Actions": metric.illegal_actions,
                "Mean TD Error": metric.mean_td_error(),
                "Unique States": metric.unique_states(),
            })
        final_reward = metrics[-1].cumulative_reward

        if save_policy and final_reward > best_reward:
            best_reward = final_reward
            best_q_table = agent.q_table.copy()

        env.close()

    dataframe = pd.DataFrame(results)

    filename = reward_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("=", "_")
    dataframe.to_csv(f"results/csv/{filename}.csv", index=False)
    config.NUM_EPISODES = old_num_episodes

    last_metric = results[-1]

    print(f"{reward_name} completed.")
    print(f"Final reward: {last_metric['Reward']:.2f}")
    print(f"Final length: {last_metric['Episode Length']}")
    print(f"Final success: {last_metric['Success']}")

    if save_policy:
        filename = reward_name.lower().replace(" ", "_") + ".npy"
        os.makedirs("results/policies", exist_ok=True)
        np.save(os.path.join("results", "policies", filename), best_q_table,) 

    return dataframe
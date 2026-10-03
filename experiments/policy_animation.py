import os
import numpy as np
import imageio.v2 as imageio
import cv2
from src.environment import create_taxi_env


def record_policy(q_table_file, output_file):

    env = create_taxi_env(render_mode="rgb_array")
    q_table = np.load(q_table_file)
    state, _ = env.reset()
    frames = []
    total_reward = 0
    step = 0
    action_names = ["South", "North", "East",
                    "West", "Pickup", "Dropoff",]

    while True:

        frame = env.render().copy()
        info_height = 70
        panel = np.full(
            (frame.shape[0] + info_height, frame.shape[1], 3), 255, dtype=np.uint8,)

        panel[:frame.shape[0], :] = frame
        frame = panel

        y = frame.shape[0] - 45

        action = int(np.argmax(q_table[state]))

        taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(state)
        passenger_names = ["Red", "Green", "Yellow", "Blue", "In Taxi",]
        destination_names = ["Red", "Green", "Yellow", "Blue",]

        if passenger == 4:
            passenger_text = "Passenger: In Taxi"
        else:
            passenger_text = f"Passenger: Waiting at {passenger_names[passenger]}"

        cv2.putText(frame, passenger_text, (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        cv2.putText(frame, f"Destination: {destination_names[destination]}", (250, y + 22),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        
        cv2.putText(frame, f"Step: {step}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        cv2.putText(frame, f"Reward: {total_reward}", (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        cv2.putText(frame, f"Action: {action_names[action]}", (300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)

        frames.append(frame)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        if terminated or truncated:

            final_frame = env.render().copy()
            panel = np.full((final_frame.shape[0] + info_height, final_frame.shape[1], 3),255,dtype=np.uint8,)
            panel[:final_frame.shape[0], :] = final_frame
            final_frame = panel

            cv2.putText(final_frame, "Episode Finished", (10, final_frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2,)
            cv2.putText(final_frame, f"Total Reward: {total_reward}", (10, final_frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
            cv2.putText(final_frame, f"Steps: {step}", (220, final_frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
            for _ in range(15):
                frames.append(final_frame)

            break

    env.close()

    imageio.mimsave( output_file, frames, fps=5, loop=0,)


def record_training_episode(action_file, initial_state_file, output_file):

    env = create_taxi_env(render_mode="rgb_array")
    actions = np.load(action_file)
    state, _ = env.reset()

    initial_state = int(np.load(initial_state_file))
    env.unwrapped.s = initial_state
    state = initial_state

    frames = []
    total_reward = 0
    step = 0
    action_names = ["South", "North", "East",
                    "West", "Pickup", "Dropoff",]

    for action in actions:

        frame = env.render().copy()
        info_height = 70
        panel = np.full(
            (frame.shape[0] + info_height, frame.shape[1], 3), 255, dtype=np.uint8,)

        panel[:frame.shape[0], :] = frame
        frame = panel

        y = frame.shape[0] - 45

        taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(state)
        passenger_names = ["Red", "Green", "Yellow", "Blue", "In Taxi",]
        destination_names = ["Red", "Green", "Yellow", "Blue",]

        if passenger == 4:
            passenger_text = "Passenger: In Taxi"
        else:
            passenger_text = f"Passenger: Waiting at {passenger_names[passenger]}"

        cv2.putText(frame, passenger_text, (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        cv2.putText(frame, f"Destination: {destination_names[destination]}", (250, y + 22),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        
        cv2.putText(frame, f"Step: {step}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        cv2.putText(frame, f"Reward: {total_reward}", (140, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
        cv2.putText(frame, f"Action: {action_names[action]}", (300, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)

        frames.append(frame)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        step += 1
        if terminated or truncated:

            final_frame = env.render().copy()
            panel = np.full((final_frame.shape[0] + info_height, final_frame.shape[1], 3),255,dtype=np.uint8,)
            panel[:final_frame.shape[0], :] = final_frame
            final_frame = panel

            cv2.putText(final_frame, "Episode Finished", (10, final_frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2,)
            cv2.putText(final_frame, f"Total Reward: {total_reward}", (10, final_frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
            cv2.putText(final_frame, f"Steps: {step}", (220, final_frame.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,)
            for _ in range(15):
                frames.append(final_frame)

            break

    env.close()

    imageio.mimsave( output_file, frames, fps=5, loop=0,)


def record_evolution():

    checkpoints = [100, 193, 301, 418, 507]
    frames = []

    for checkpoint in checkpoints:

        temp_file = "results/animations/evolution/temp.gif"
        record_training_episode(f"results/policies/checkpoints/actions_{checkpoint}.npy",
             f"results/policies/checkpoints/initial_state_{checkpoint}.npy", temp_file,)

        gif = imageio.mimread(temp_file)
        title_frame = gif[0].copy()

        cv2.putText(title_frame, f"Policy after {checkpoint} Training Episodes", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,)

        frames.extend([title_frame] * 10)
        frames.extend(gif)

    imageio.mimsave("results/animations/evolution/policy_evolution.gif", frames, fps=5, loop=0,)

    os.remove(temp_file)


def main():

    os.makedirs("results/animations/best_policies", exist_ok=True)
    os.makedirs("results/animations/evolution", exist_ok=True)
    policies = {
        "base_reward_sarsa": "base_reward_sarsa.npy",
        "base_reward_planning": "base_reward_planning.npy",
        "base_reward_kernel": "base_reward_kernel.npy",
        "base_reward_full": "base_reward_full.npy",

        "custom_reward_1_sarsa": "custom_reward_1_sarsa.npy",
        "custom_reward_1_planning": "custom_reward_1_planning.npy",
        "custom_reward_1_kernel": "custom_reward_1_kernel.npy",
        "custom_reward_1_full": "custom_reward_1_full.npy",
    }

    for name, file in policies.items():

        print(f"Generating {name}...")

        record_policy(
            f"results/policies/{file}",
            f"results/animations/best_policies/{name}.gif",
        )
    record_evolution()


if __name__ == "__main__":
    main()
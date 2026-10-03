def base_reward(env, state, action, next_state, env_reward, terminated, step_count):  # Use the original Taxi reward
    return env_reward

def sparse_reward(env, state, action, next_state, env_reward, terminated, step_count):
    # Reward successful pickup and successful dropoff.

    _, _, passenger_before, _ = env.unwrapped.decode(state)
    _, _, passenger_after, _ = env.unwrapped.decode(next_state)

    if terminated:
        return 0.5

    if passenger_before != 4 and passenger_after == 4:
        return 0.5

    return 0

def custom_reward_1(env, state, action, next_state, env_reward, terminated, step_count):
    # Fast Delivery Reward

    if env_reward == -10:
        return -10

    if terminated:
        bonus = max(0, 20 - (step_count // 10))
        return 20 + bonus

    return -1

def custom_reward_2(env, state, action, next_state, env_reward, terminated, step_count):
    # Efficient Driving Reward

    if env_reward == -10:
        return -15

    if terminated:
        return 30

    # Extra penalty for trying to move into a wall.
    if action in [0, 1, 2, 3] and state == next_state:
        return -3

    return -1

def custom_reward_3(env, state, action, next_state, env_reward, terminated, step_count):
    # Progress Reward

    taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(state)
    next_row, next_col, next_passenger, _ = env.unwrapped.decode(next_state)

    locations = env.unwrapped.locs

    if env_reward == -10:
        return -10

    if terminated:
        return 30

    # Before pickup: encourage moving toward the passenger
    if passenger != 4:

        passenger_row, passenger_col = locations[passenger]

        before = abs(taxi_row - passenger_row) + abs(taxi_col - passenger_col)
        after = abs(next_row - passenger_row) + abs(next_col - passenger_col)

    # After pickup: encourage moving toward the destination
    else:

        destination_row, destination_col = locations[destination]

        before = abs(taxi_row - destination_row) + abs(taxi_col - destination_col)
        after = abs(next_row - destination_row) + abs(next_col - destination_col)

    if after < before:
        return 1

    if after > before:
        return -2

    return -1
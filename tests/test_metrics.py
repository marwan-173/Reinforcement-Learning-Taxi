from src.metrics import Metrics


# Test the Metrics class.

metrics = Metrics()

metrics.add_reward(10)
metrics.add_reward(-1)

metrics.add_step()
metrics.add_step()

metrics.add_state(5)
metrics.add_state(10)
metrics.add_state(5)

metrics.add_td_error(0.5)
metrics.add_td_error(-1.5)

metrics.add_illegal_action()
metrics.set_success()

assert metrics.cumulative_reward == 9
assert metrics.episode_length == 2
assert metrics.unique_states() == 2
assert metrics.illegal_actions == 1
assert metrics.success
assert metrics.mean_td_error() == 1.0

print("Metrics test passed.")
from src.environment import create_taxi_env
from src.kernel import Kernel
import src.config as config

# Test the Kernel similarity function.

env = create_taxi_env()

kernel = Kernel(env, config)

state1 = env.reset()[0]
state2 = state1

similarity = kernel.similarity(state1, state2)

assert 0.0 <= similarity <= 1.0
assert similarity > 0.9

print("Kernel test passed.")
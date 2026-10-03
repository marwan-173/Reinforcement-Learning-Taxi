"""
Project configuration.
"""


# Environment
SEED = 42
MAX_EPISODE_STEPS = 200


# Training
NUM_EPISODES = 1000
NUM_SEEDS = 5


# SARSA
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.1


# Prioritized Sweeping
PLANNING_STEPS = 5


# Generalization Kernel
KERNEL_BETA = 4.0
KERNEL_C = 0.5
KERNEL_NEIGHBORS = 5


# Evaluation
NUM_EVALUATION_EPISODES = 100
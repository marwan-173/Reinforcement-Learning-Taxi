import numpy as np


class Kernel:

    def __init__(self, env, config):      #Initializing similarity kernel.
        self.env = env
        self.config = config
        self.neighbors = {}
        self.build_neighbors()

    def build_neighbors(self):
    # Precompute the closest states for every Taxi state.

        for state in range(self.env.observation_space.n):

            taxi_row, taxi_col, passenger, destination = \
                self.env.unwrapped.decode(state)

            candidates = []
            for other_state in range(self.env.observation_space.n):
                if state == other_state:
                    continue

                other_row, other_col, other_passenger, other_destination = \
                    self.env.unwrapped.decode(other_state)
                if passenger != other_passenger or destination != other_destination:
                    continue

                distance = abs(taxi_row - other_row) + abs(taxi_col - other_col)
                candidates.append((distance, other_state))

            candidates.sort(key=lambda x: x[0])
            self.neighbors[state] = [
                s for _, s in candidates[:self.config.KERNEL_NEIGHBORS]
            ]
    
    def get_neighbors(self, state):
        return self.neighbors[state]
    
    def similarity(self, state1, state2):      #Compute similarity between two Taxi states.

        taxi_row1, taxi_col1, passenger1, destination1 = self.env.unwrapped.decode(state1)
        taxi_row2, taxi_col2, passenger2, destination2 = self.env.unwrapped.decode(state2)

        # States with different passenger status or destination are not similar.
        if passenger1 != passenger2 or destination1 != destination2:
            return 0.0

        distance = abs(taxi_row1 - taxi_row2) + abs(taxi_col1 - taxi_col2)

        beta = self.config.KERNEL_BETA
        c = self.config.KERNEL_C

        return 1.0 / (1.0 + np.exp(beta * (distance - c)))
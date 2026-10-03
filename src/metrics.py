class Metrics:

    def __init__(self):      #Initializing episode metrics.
        self.reset()

    def reset(self):      #Reset all metrics for a new episode.
        self.cumulative_reward = 0.0
        self.episode_length = 0
        self.illegal_actions = 0
        self.td_error_sum = 0.0
        self.td_updates = 0
        self.success = False
        self.visited_states = set()


    def add_reward(self, reward):      #Add reward to the cumulative reward.
        self.cumulative_reward += reward

    def add_step(self):      #Increase the episode length.
        self.episode_length += 1

    def add_state(self, state):      #Store a visited state.
        self.visited_states.add(state)

    def add_td_error(self, td_error):      #Store the absolute TD error.
        self.td_error_sum += abs(td_error)
        self.td_updates += 1

    def add_illegal_action(self):      #Increase the illegal action counter.
        self.illegal_actions += 1

    def set_success(self):      #Mark the episode as successful.
        self.success = True

    def mean_td_error(self):      #Return the mean absolute TD error.

        if self.td_updates == 0:
            return 0.0
        return self.td_error_sum / self.td_updates

    def unique_states(self):      #Return the number of unique visited states.
        return len(self.visited_states)
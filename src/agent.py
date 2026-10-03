import os
import numpy as np
from tqdm import tqdm
from src.metrics import Metrics
from src.planning import Planner
from src.kernel import Kernel

class Agent:

    def __init__(self, env, reward_function, config, use_planning=False, use_kernel=False): #initializing agent
        self.env = env
        self.reward_function = reward_function
        self.config = config
        self.use_planning = use_planning
        self.use_kernel = use_kernel

        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=float)
        self.model = {}      #Transition model for planning.
        self.predecessors = {}      #Predecessors for Prioritized Sweeping.
        self.planner = Planner()
        self.kernel = Kernel(env, config)

    def greedy_action(self, state):                 #return the greedy action for a state.
        return int(np.argmax(self.q_table[state]))


    def choose_action(self, state):                #Epsilon-greedy action selection.
        if np.random.random() < self.config.EPSILON:
            return self.env.action_space.sample()
        return self.greedy_action(state)


    def sarsa_update(self, state, action, reward, next_state, next_action, terminated):   #Perform one SARSA update.
        current_q = self.q_table[state, action]

        if terminated:
            target = reward
        else:
            target = reward + (self.config.GAMMA * self.q_table[next_state, next_action])

        td_error = target - current_q
        self.q_table[state, action] += (self.config.ALPHA * td_error)

        return td_error


    def store_transition(self, state, action, reward, next_state, terminated):      #Save transition for future planning.

        self.model[(state, action)] = (reward, next_state, terminated)

        if next_state not in self.predecessors:
            self.predecessors[next_state] = set()

        self.predecessors[next_state].add((state, action))


    def compute_priority(self, state, action, reward, next_state, terminated):   #Compute the priority of a transition.

        current_q = self.q_table[state, action]

        if terminated:
            target = reward
        else:
            next_action = self.choose_action(next_state)
            target = reward + (self.config.GAMMA * self.q_table[next_state, next_action])

        return abs(target - current_q)


    def planning_step(self):      #Perform prioritized sweeping planning.

        for _ in range(self.config.PLANNING_STEPS):

            if self.planner.is_empty():
                break

            _, state, action = self.planner.pop()

            reward, next_state, terminated = self.model[(state, action)]

            if terminated:
                next_action = 0
            else:
                next_action = self.choose_action(next_state)

            td_error = self.sarsa_update(state, action, reward, next_state, next_action, terminated)
            if next_state in self.predecessors:
                for predecessor_state, predecessor_action in self.predecessors[next_state]:
                    predecessor_reward, predecessor_next_state, predecessor_terminated = self.model[
                        (predecessor_state, predecessor_action)]

                    priority = self.compute_priority(predecessor_state, predecessor_action,
                        predecessor_reward, predecessor_next_state, predecessor_terminated)

                    self.planner.add(priority,predecessor_state,predecessor_action)


    def kernel_update(self, state, action, td_error):      #Generalize an update to similar states.

        for similar_state in self.kernel.get_neighbors(state):
            similarity = self.kernel.similarity(state, similar_state)
            if similarity == 0:
                continue

            self.q_table[similar_state, action] += (self.config.ALPHA * similarity * td_error)


    def process_transition(self, state, action, reward, env_reward, next_state, next_action, terminated, metrics):   #Process one transition.

        td_error = self.sarsa_update(state, action, reward, next_state, next_action, terminated)
        if self.use_kernel:
            self.kernel_update(state, action, td_error)

        self.store_transition(state, action, reward, next_state, terminated)
        if self.use_planning:
            priority = self.compute_priority(state, action, reward, next_state, terminated)
            self.planner.add(priority, state, action)
            self.planning_step()

        metrics.add_reward(reward)
        metrics.add_step()
        metrics.add_state(state)
        metrics.add_td_error(td_error)

        if env_reward == -10:
            metrics.add_illegal_action()

        if terminated:
            metrics.set_success()

    def train_episode(self):      #Train the agent for one episode.

        state, _ = self.env.reset()
        initial_state = state
        action = self.choose_action(state)

        metrics = Metrics()
        actions = []

        while True:

            actions.append(action)
            next_state, env_reward, terminated, truncated, _ = self.env.step(action)
            reward = self.reward_function(self.env, state, action, next_state, env_reward, terminated, metrics.episode_length)
            if self.reward_function.__name__ == "sparse_reward" and reward > 1:
                print(
                    f"Impossible reward detected: reward={reward}, "
                    f"env_reward={env_reward}, "
                    f"terminated={terminated}"
                )
            if terminated or truncated:
                next_action = 0
            else:
                next_action = self.choose_action(next_state)

            self.process_transition(state, action, reward, env_reward, next_state, next_action, terminated, metrics)

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        return metrics, actions, initial_state


    def train(self, description="Training", save_checkpoints=False):      #Train the agent for multiple episodes.

        episode_metrics = []

        for episode in tqdm(range(self.config.NUM_EPISODES), desc=description, leave=False,):
            metrics, actions, initial_state = self.train_episode()
            episode_metrics.append(metrics)
            if save_checkpoints:

                checkpoints = {100, 193, 301, 418, 507, 1000}
                current_episode = episode + 1
                if current_episode in checkpoints:

                    os.makedirs("results/policies/checkpoints", exist_ok=True)
                    np.save(f"results/policies/checkpoints/episode_{current_episode}.npy", self.q_table,)
                    np.save(f"results/policies/checkpoints/actions_{current_episode}.npy", np.array(actions, dtype=np.int32),)
                    np.save(f"results/policies/checkpoints/initial_state_{current_episode}.npy", np.array(initial_state),)
        return episode_metrics
    

    def save_q_table(self, filename):
        os.makedirs("results/policies", exist_ok=True)
        np.save(os.path.join("results", "policies", filename), self.q_table,)
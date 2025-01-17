import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = list(range(env.action_space.n))  # List of possible actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Initialize the Q-table
        self.state_bins = self.create_bins(env.observation_space)

        self.word_q_table = {}  # Q-table for word difficulties

    def create_bins(self, observation_space, num_bins=10):
        bins = []
        for i in range(len(observation_space.high)):
            high = observation_space.high[i]
            low = observation_space.low[i]
            bins.append(np.linspace(low, high, num_bins))
        return bins

    def discretize_state(self, state):
        state_indices = []
        for i in range(len(state)):
            state_indices.append(np.digitize(state[i], self.state_bins[i]) - 1)
        return tuple(state_indices)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = self.get_best_action(state)
        return action

    def get_best_action(self, state):
        state = self.discretize_state(state)
        self.q_table.setdefault(state, {action: 0 for action in self.actions})
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_table(self, state, action, reward, next_state):
        print(state, action, reward, next_state)
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        self.q_table.setdefault(state, {action: 0 for action in self.actions})
        self.q_table.setdefault(next_state, {action: 0 for action in self.actions})
        best_next_action = self.get_best_action(next_state)
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action]
        )

    def adjust_word_difficulty(self, word, reward):
        self.word_q_table.setdefault(word, 0)
        self.word_q_table[word] += self.alpha * (reward - self.word_q_table[word])

    def choose_word(self):
        if not self.word_q_table:
            return random.choice(["apple", "banana", "cat", "dog", "elephant"])
        return max(self.word_q_table, key=self.word_q_table.get)

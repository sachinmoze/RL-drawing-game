import numpy as np
import random
import gym
import matplotlib.pyplot as plt
import pandas as pd

def create_bins(observation_space, num_bins=10):
  bins = []
  for i in range(len(observation_space.high)):
    high = observation_space.high[i]
    low = observation_space.low[i]
    bins.append(pd.cut(range(num_bins), num_bins, labels=False) + low)
  return bins

def discretize_state(state, bins):
    state_indices = []
    for i in range(len(state)):
        state_indices.append(np.digitize(state[i], bins[i]) - 1)
    return tuple(state_indices)

def process_state(state):
    if isinstance(state, dict):
      state = np.asarray([state[key] for key in sorted(state.keys())])
    else:
      state = np.asarray(state)
    return state

def q_learning(env, num_episodes, alpha, gamma, epsilon, num_bins):
    bins = create_bins(env.observation_space, num_bins)
    q_table = {}
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = process_state(state)
        state = discretize_state(state, bins)
        total_reward = 0
        done = False

        while not done:
            if state not in q_table:
                q_table[state] = {action: 0 for action in range(env.action_space.n)}

            if np.random.uniform(0, 1) < epsilon:
                action = random.choice(list(q_table[state].keys()))
            else:
                action = max(q_table[state], key=q_table[state].get)

            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            next_state = discretize_state(next_state, bins)

            if next_state not in q_table:
                q_table[next_state] = {action: 0 for action in range(env.action_space.n)}

            best_next_action = max(q_table[next_state], key=q_table[next_state].get)
            q_table[state][action] += alpha * (
                reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
            )

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    return q_table, rewards

def monte_carlo(env, num_episodes, gamma, epsilon, num_bins):
    bins = create_bins(env.observation_space, num_bins)
    returns_sum = {}
    returns_count = {}
    q_table = {}
    rewards = []

    for episode in range(num_episodes):
        episode_states = []
        episode_actions = []
        episode_rewards = []

        state = env.reset()
        state = process_state(state)
        state = discretize_state(state, bins)
        done = False

        while not done:
            if state not in q_table:
                q_table[state] = {action: 0 for action in range(env.action_space.n)}

            if np.random.uniform(0, 1) < epsilon:
                action = random.choice(list(q_table[state].keys()))
            else:
                action = max(q_table[state], key=q_table[state].get)

            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            next_state = discretize_state(next_state, bins)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state

        G = 0
        for t in range(len(episode_states) - 1, -1, -1):
            state = episode_states[t]
            action = episode_actions[t]
            reward = episode_rewards[t]

            G = gamma * G + reward

            if state not in episode_states[:t] or action not in episode_actions[:t]:
                if (state, action) not in returns_sum:
                    returns_sum[(state, action)] = 0
                    returns_count[(state, action)] = 0
                returns_sum[(state, action)] += G
                returns_count[(state, action)] += 1
                q_table[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

        rewards.append(sum(episode_rewards))

    return q_table, rewards

def evaluate_policy(env, Q, num_episodes, num_bins):
    bins = create_bins(env.observation_space, num_bins)
    total_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        state = process_state(state)
        state = discretize_state(state, bins)
        total_reward = 0
        done = False

        while not done:
            if state in Q:
                action = max(Q[state], key=Q[state].get)
            else:
                action = env.action_space.sample()

            state, reward, done, _ = env.step(action)
            state = process_state(state)
            state = discretize_state(state, bins)
            total_reward += reward

        total_rewards.append(total_reward)

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    return avg_reward, std_reward

env = gym.make('CartPole-v1')
num_episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_bins = 10

print("Starting Q-learning...")
Q_q_learning, rewards_q_learning = q_learning(env, num_episodes, alpha, gamma, epsilon, num_bins)
print("Q-learning complete!")

print("Starting Monte Carlo...")
Q_monte_carlo, rewards_monte_carlo = monte_carlo(env, num_episodes, gamma, epsilon, num_bins)
print("Monte Carlo complete!")

print("Evaluating Q-learning policy...")
avg_reward_q_learning, std_reward_q_learning = evaluate_policy(env, Q_q_learning, 100, num_bins)
print("Evaluating Monte Carlo policy...")
avg_reward_monte_carlo, std_reward_monte_carlo = evaluate_policy(env, Q_monte_carlo, 100, num_bins)

# Plotting the results
methods = ['Q-learning', 'Monte Carlo']
avg_rewards = [avg_reward_q_learning, avg_reward_monte_carlo]
std_rewards = [std_reward_q_learning, std_reward_monte_carlo]

plt.figure(figsize=(10, 6))
plt.bar(methods, avg_rewards, yerr=std_rewards, capsize=5)
plt.ylabel('Average Reward', fontsize=12, fontname='Arial')
plt.title('Comparison of Q-learning and Monte Carlo on CartPole-v1', fontsize=12, fontname='Arial')
plt.xticks(fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')
plt.show()

# Plotting reward vs. episodes
plt.figure(figsize=(10, 6))
plt.plot(rewards_q_learning, label='Q-learning', color='blue')
plt.plot(rewards_monte_carlo, label='Monte Carlo', color='green')
plt.xlabel('Episodes', fontsize=12, fontname='Arial')
plt.ylabel('Reward', fontsize=12, fontname='Arial')
plt.title('Reward vs Episodes', fontsize=12, fontname='Arial')
plt.legend()
plt.show()

# Display results in a table
results = pd.DataFrame({
    'Method': methods,
    'Average Reward': avg_rewards,
    'Standard Deviation': std_rewards
})
print(results)

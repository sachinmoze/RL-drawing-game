import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Initialize the CartPole-v1 environment
env = gym.make('CartPole-v1')

def create_bins(num_bins, env):
    """Create bins for discretizing the state space."""
    bins = []
    for i in range(len(env.observation_space.low)):
        low, high = env.observation_space.low[i], env.observation_space.high[i]
        bins.append(np.linspace(low, high, num_bins - 1))
    return bins

def discretize_state(state, bins):
    """Discretize the continuous state space into discrete bins."""
    return tuple(np.digitize(s, b) for s, b in zip(state, bins))

def initialize_q(num_bins, action_space_n):
    """Initialize the Q-table with zeros."""
    Q = {}
    for indices in np.ndindex(*(num_bins + 1,) * 4):
        Q[indices] = np.zeros(action_space_n)
    return Q

def initialize_returns(num_bins, action_space_n):
    """Initialize the returns dictionary."""
    returns = {}
    for indices in np.ndindex(*(num_bins + 1,) * 4):
        for action in range(action_space_n):
            returns[(indices, action)] = []
    return returns

def epsilon_greedy(Q, state, epsilon, nA):
    """Epsilon-greedy policy for action selection."""
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    else:
        return np.argmax(Q[state])

def monte_carlo(env, num_episodes, gamma, epsilon, num_bins):
    """Monte Carlo algorithm for learning Q-values."""
    bins = create_bins(num_bins, env)
    Q = initialize_q(num_bins + 1, env.action_space.n)
    returns = initialize_returns(num_bins + 1, env.action_space.n)
    rewards_per_episode = []
    for i in range(num_episodes):
        if i % 100 == 0:
            print(f"Monte Carlo Episode: {i}")
        episode = []
        state, _ = env.reset()
        state = discretize_state(state, bins)
        done = False
        total_reward = 0
        while not done:
            action = epsilon_greedy(Q, state, epsilon, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)
        G = 0
        visited = set()
        for state, action, reward in reversed(episode):
            G = gamma * G + reward
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
    return Q, rewards_per_episode

def q_learning(env, num_episodes, alpha, gamma, epsilon, num_bins):
    """Q-learning algorithm for learning Q-values."""
    bins = create_bins(num_bins, env)
    Q = initialize_q(num_bins + 1, env.action_space.n)
    rewards_per_episode = []
    for i in range(num_episodes):
        if i % 100 == 0:
            print(f"Q-learning Episode: {i}")
        state, _ = env.reset()
        state = discretize_state(state, bins)
        done = False
        total_reward = 0
        while not done:
            action = epsilon_greedy(Q, state, epsilon, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, bins)
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)
    return Q, rewards_per_episode

def evaluate_policy(env, Q, num_episodes, num_bins):
    """Evaluate the policy by running it for a number of episodes and calculating the average reward."""
    bins = create_bins(num_bins, env)
    total_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = discretize_state(state, bins)
        total_reward = 0
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _, _ = env.step(action)
            state = discretize_state(state, bins)
            total_reward += reward
        total_rewards.append(total_reward)
    return np.mean(total_rewards), np.std(total_rewards)

def main():
    num_episodes = 1000
    alpha = 0.1
    gamma = 0.99
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

if __name__ == '__main__':
    main()

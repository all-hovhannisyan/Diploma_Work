import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map

# Set up the environment (compatible with gym 0.26.0+)
try:
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=True,
        render_mode="human",  # Enable graphical rendering
        desc=None  # Use default 4x4 map (or pass generate_random_map(size=4))
    )
    print("Environment loaded successfully!")
except Exception as e:
    print(f"Error creating environment: {e}")
    exit()

# Initialize Q-table
state_size = env.observation_space.n
action_size = env.action_space.n
Q = np.zeros((state_size, action_size))

# Hyperparameters
alpha = 0.8       # Learning rate
gamma = 0.95      # Discount factor
epsilon = 1.0     # Exploration rate (starts at 100%)
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 2000   # Training episodes
max_steps = 100   # Steps per episode

# Track rewards
rewards_per_episode = []

# Training loop (no rendering for speed)
for episode in range(episodes):
    state, _ = env.reset()  # New gym API returns (state, info)
    total_reward = 0
    done = False

    for _ in range(max_steps):
        # Epsilon-greedy action
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state, :])    # Exploit

        # Take action
        next_state, reward, done, truncated, info = env.step(action)

        # Update Q-table (Bellman equation)
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_per_episode.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon

    # Print progress
    if episode % 500 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(rewards_per_episode, alpha=0.3, label="Raw")
plt.plot(
    np.convolve(rewards_per_episode, np.ones(100)/100, mode="valid"),
    label="Smoothed (100-episode avg)"
)
plt.title("Q-Learning Progress (FrozenLake-v1)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.grid()
plt.show()

# Test the trained agent
print("\n=== Testing Policy ===")
test_episodes = 3

for episode in range(test_episodes):
    state, _ = env.reset()
    done = False
    print(f"\nTest Episode {episode + 1}")

    for step in range(max_steps):
        action = np.argmax(Q[state, :])  # Greedy policy
        next_state, reward, done, truncated, info = env.step(action)
        env.render()  # Show graphical output
        state = next_state

        if done:
            print(f"Result: {'Goal reached! 🎉' if reward == 1 else 'Fell in a hole! 💀'}")
            break

env.close()
import numpy as np
import gym
import matplotlib.pyplot as plt

# Ստեղծում ենք ուսուցման միջավայրը
train_env = gym.make("FrozenLake-v1", is_slippery=True)  # No render_mode here

# Q-աղյուսակի սկզբնարժեքավորումը
state_size = train_env.observation_space.n
action_size = train_env.action_space.n
Q = np.zeros((state_size, action_size))

# Հիպերպարամետրերը
alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 2000
max_steps = 100

rewards_per_episode = []

# Ուսուցումը
for episode in range(episodes):
    state, _ = train_env.reset()
    total_reward = 0
    done = False

    for step in range(max_steps):
        # Epsilon-greedy
        if np.random.random() < epsilon:
            action = train_env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state, reward, done, truncated, _ = train_env.step(action)
    

        # Q-update
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state, :]) - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if done:
            break

    rewards_per_episode.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

train_env.close()

# Արդյունքների առաջընթացի գրաֆիկը
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


print("\n Testing Trained Agent")

test_env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
test_episodes = 3

for episode in range(test_episodes):
    state, _ = test_env.reset()
    done = False
    print(f"\nTest Episode {episode + 1}")

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        next_state, reward, done, truncated, _ = test_env.step(action)
        state = next_state

        if done:
            print(f"Result: {'Goal reached!' if reward == 1 else 'Fell in a hole!'}")
            break

test_env.close()



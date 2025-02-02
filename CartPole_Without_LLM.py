import gymnasium as gym
import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

# --------------------
#  HYPERPARAMETERS
# --------------------

NUM_BUCKETS = (10, 10, 20, 20)
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 500
ALPHA = 0.05
GAMMA = 0.99
EPSILON_INITIAL = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9995
ANGLE_PENALTY_COEFF = 0.1

# --------------------
#  ENVIRONMENTS
# --------------------

env_train = gym.make("CartPole-v1")
env_watch = gym.make("CartPole-v1", render_mode='human')

# --------------------
#  DISCRETIZATION
# --------------------

STATE_BOUNDS = list(zip(env_train.observation_space.low, env_train.observation_space.high))
STATE_BOUNDS[0] = (-4.8, 4.8)  
STATE_BOUNDS[1] = (-3.0, 3.0)  
STATE_BOUNDS[2] = (-0.418, 0.418)
STATE_BOUNDS[3] = (-4.0, 4.0)

def discretize_state(state):
    bins = []
    for i in range(len(state)):
        low, high = STATE_BOUNDS[i]
        val = min(max(state[i], low), high)
        ratio = (val - low) / (high - low)
        bucket_idx = int(round((NUM_BUCKETS[i] - 1) * ratio))
        bins.append(bucket_idx)
    return tuple(bins)

# --------------------
#  Q-TABLE
# --------------------

Q_table = np.zeros(NUM_BUCKETS + (env_train.action_space.n,))

# --------------------
#  POLICY (EPS-GREEDY)
# --------------------

def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, env_train.action_space.n)
    else:
        return np.argmax(Q_table[state])

def update_q_table(state, action, reward, next_state, alpha, gamma):
    best_next_action = np.argmax(Q_table[next_state])
    td_target = reward + gamma * Q_table[next_state][best_next_action]
    td_error = td_target - Q_table[state][action]
    Q_table[state][action] += alpha * td_error

# --------------------
#  TRAINING LOOP
# --------------------

def train():
    global EPSILON_INITIAL
    scores = []

    for episode in range(NUM_EPISODES):
        state, _ = env_train.reset()
        state = discretize_state(state)
        done = False
        truncated = False
        total_reward = 0

        epsilon = max(EPSILON_MIN, EPSILON_INITIAL * (EPSILON_DECAY ** episode))

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = choose_action(state, epsilon)
            next_obs, reward, done, truncated, info = env_train.step(action)

            if ANGLE_PENALTY_COEFF != 0:
                reward -= abs(next_obs[2]) * ANGLE_PENALTY_COEFF

            next_state = discretize_state(next_obs)
            update_q_table(state, action, reward, next_state, ALPHA, GAMMA)

            state = next_state
            total_reward += reward

            if done or truncated:
                break

        scores.append(total_reward)
        # Print if desired
        # print(f"Episode {episode+1}/{NUM_EPISODES} - Reward: {total_reward}")

    return scores

# --------------------
#  MAIN
# --------------------

if __name__ == "__main__":
    start_time = time.time()

    print("Starting Q-learning training on CartPole...")
    scores = train()

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

    # Save the Q-table if desired
    q_path = os.path.join(".", "q_table_cartpole.json")
    with open(q_path, "w") as f:
        json.dump(Q_table.tolist(), f)
    print(f"Q-table saved to {q_path}")

    # --------------------
    #  PLOT ONLY FIRST 50 EPISODES
    # --------------------
    # plt.figure(figsize=(10, 6))
    # plt.plot(scores[:50], label='Reward per Episode (First 50)')
    # plt.title('Training Rewards (Episodes 1-50)')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.legend()
    # plt.show()

    # --------------------
    #  WATCH THE AGENT
    # --------------------
    print("\nWatching the trained agent for a few episodes...")
    test_episodes = 5
    for ep in range(test_episodes):
        obs, _ = env_watch.reset()
        state = discretize_state(obs)
        done = False
        truncated = False
        total_reward = 0

        while not done and not truncated:
            action = np.argmax(Q_table[state])
            obs, reward, done, truncated, info = env_watch.step(action)
            env_watch.render()
            total_reward += reward
            state = discretize_state(obs)
            time.sleep(0.02)

        print(f"Test Episode {ep+1}: Total Reward = {total_reward}")

    env_train.close()
    env_watch.close()

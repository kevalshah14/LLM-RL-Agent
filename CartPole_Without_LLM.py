import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Hyperparameters
# ----------------------------------------------------
ENV_NAME = "CartPole-v1"
N_ITERATIONS = 12        # Number of CEM iterations
BATCH_SIZE = 25          # Number of parameter vectors sampled per iteration
ELITE_FRAC = 0.2         # Fraction of top samples used to update distribution
HIDDEN_SIZE = 8          # Hidden layer size
RANDOM_SEED = 42         # For reproducibility (optional)

# => 12 * 25 = 300 episodes total

# ----------------------------------------------------
# 2. Create the Environment (for training)
# ----------------------------------------------------
env = gym.make(ENV_NAME)
np.random.seed(RANDOM_SEED)
env.action_space.seed(RANDOM_SEED)

# ----------------------------------------------------
# 3. Define Our Policy Network in NumPy
# ----------------------------------------------------
# We'll define a single hidden-layer NN: 4 inputs -> HIDDEN_SIZE -> 2 outputs
# We'll store all parameters as a single 1D vector [W1, b1, W2, b2],
# shaped as follows:
#   W1: shape (4, HIDDEN_SIZE)
#   b1: shape (HIDDEN_SIZE,)
#   W2: shape (HIDDEN_SIZE, 2)
#   b2: shape (2,)

def get_param_sizes():
    """Return the shapes (and total size) of each parameter array."""
    w1_shape = (4, HIDDEN_SIZE)
    b1_shape = (HIDDEN_SIZE,)
    w2_shape = (HIDDEN_SIZE, 2)
    b2_shape = (2,)
    
    shapes = [w1_shape, b1_shape, w2_shape, b2_shape]
    total_params = sum(np.prod(s) for s in shapes)
    return shapes, total_params

SHAPES, PARAM_SIZE = get_param_sizes()

def unflatten_params(theta):
    """
    Turn a 1D parameter vector into (W1, b1, W2, b2) with correct shapes.
    """
    idx = 0
    w1_size = np.prod(SHAPES[0])  # 4 * HIDDEN_SIZE
    w1 = theta[idx : idx + w1_size].reshape(SHAPES[0])
    idx += w1_size
    
    b1_size = np.prod(SHAPES[1])  # HIDDEN_SIZE
    b1 = theta[idx : idx + b1_size].reshape(SHAPES[1])
    idx += b1_size
    
    w2_size = np.prod(SHAPES[2])  # HIDDEN_SIZE * 2
    w2 = theta[idx : idx + w2_size].reshape(SHAPES[2])
    idx += w2_size
    
    b2_size = np.prod(SHAPES[3])  # 2
    b2 = theta[idx : idx + b2_size].reshape(SHAPES[3])
    idx += b2_size
    
    return w1, b1, w2, b2

def forward_pass(theta, state):
    """Compute the action probabilities given a state, using parameters theta."""
    w1, b1, w2, b2 = unflatten_params(theta)
    
    # Hidden layer: ReLU( state @ W1 + b1 )
    h = np.dot(state, w1) + b1
    h = np.maximum(h, 0)  # ReLU

    # Output layer: logits = h @ W2 + b2
    logits = np.dot(h, w2) + b2
    
    # Convert to probabilities with softmax
    max_logit = np.max(logits)  
    exps = np.exp(logits - max_logit)
    probs = exps / np.sum(exps)
    return probs  # shape (2,)

def sample_action(theta, state):
    """Sample an action (0 or 1) using the policy probabilities."""
    probs = forward_pass(theta, state)
    # Probability of action 0 is probs[0], of action 1 is probs[1].
    return 0 if np.random.rand() < probs[0] else 1

# ----------------------------------------------------
# 4. Evaluate One Episode (No Step Cap)
# ----------------------------------------------------
def evaluate_policy(theta, render=False):
    """
    Run a single episode until the environment signals done/truncated.
    Return total reward. 
    """
    obs, info = env.reset()
    total_reward = 0
    while True:
        if render:
            env.render()

        action = sample_action(theta, obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # If either done or truncated, exit
        if done or truncated:
            break
    return total_reward

# ----------------------------------------------------
# 5. Cross-Entropy Method (CEM) Loop
# ----------------------------------------------------
def cem_train(n_iter, batch_size, elite_frac=0.2):
    """
    Run Cross-Entropy Method training for n_iter iterations,
    each with batch_size episodes. Keep track of best solution.
    """
    # Initialize distribution over parameters: mean=0, std=1
    mu = np.zeros(PARAM_SIZE)
    sigma = np.ones(PARAM_SIZE)

    n_elite = int(batch_size * elite_frac)
    
    all_rewards = []
    best_score = -1e9
    best_theta = None

    for iteration in range(n_iter):
        # Sample a batch of parameter vectors from Gaussian(mu, sigma)
        thetas = np.random.randn(batch_size, PARAM_SIZE) * sigma + mu
        
        # Evaluate each parameter vector (1 episode per vector)
        rewards = np.array([evaluate_policy(t) for t in thetas])
        all_rewards.extend(rewards)
        
        # Find the elite set (top n_elite by reward)
        elite_idx = rewards.argsort()[::-1][:n_elite]
        elite_thetas = thetas[elite_idx]
        elite_rewards = rewards[elite_idx]

        # Update best if needed
        top_score = np.max(elite_rewards)
        if top_score > best_score:
            best_score = top_score
            best_theta = elite_thetas[np.argmax(elite_rewards)]

        # Refit Gaussian to elite params
        mu = elite_thetas.mean(axis=0)
        sigma = elite_thetas.std(axis=0) + 1e-8  # small epsilon to avoid zero std

        # Print iteration stats
        print(f"Iter {iteration+1}/{n_iter}, max reward={top_score:.1f}, mean elite reward={elite_rewards.mean():.1f}")

    return best_theta, all_rewards

# ----------------------------------------------------
# 6. Main
# ----------------------------------------------------
if __name__ == "__main__":
    # Train the policy using CEM
    best_params, reward_history = cem_train(N_ITERATIONS, BATCH_SIZE, ELITE_FRAC)

    # Test the best found policy over several episodes (without rendering)
    test_episodes = 5
    test_rewards = []
    for _ in range(test_episodes):
        R = evaluate_policy(best_params, render=False)
        test_rewards.append(R)
    print(f"\nTested best policy over {test_episodes} episodes. Mean reward: {np.mean(test_rewards):.1f}")

    # Close the training environment
    env.close()

    # ----------------------------------------------------
    # Demonstrate the Best Policy with Human Rendering
    # ----------------------------------------------------
    # Create a new environment with human rendering enabled
    demo_env = gym.make(ENV_NAME, render_mode="human")
    print("\nDemonstrating best policy with human render. Close the render window to finish this episode.")

    demo_total_reward = 0
    obs, info = demo_env.reset()
    while True:
        # When using render_mode="human", calling demo_env.render() is optional,
        # as rendering is handled automatically. However, it doesn't hurt to call it.
        demo_env.render()
        action = sample_action(best_params, obs)
        obs, reward, done, truncated, info = demo_env.step(action)
        demo_total_reward += reward
        if done or truncated:
            break
    print("Demo episode reward:", demo_total_reward)
    demo_env.close()

    # ----------------------------------------------------
    # Plot Reward History (Optional)
    # ----------------------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(reward_history, label="Reward per Episode (all samples)")
    plt.axhline(y=500, color='r', linestyle='--', label="Max = 500")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Cross-Entropy Method on CartPole (No Per-Episode Step Cap)")
    plt.legend()
    plt.show()

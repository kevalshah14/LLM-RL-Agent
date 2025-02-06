import gymnasium as gym
import numpy as np
from openai import OpenAI
import time
import os

# ----------------------------------------------------
# 1. Set Up and Environment
# ----------------------------------------------------
ENV_NAME = "CartPole-v1"
env = gym.make(ENV_NAME)
key = os.getenv('OPENAI_API_KEY')
# Set your OpenAI API key (replace with your actual key)
client = OpenAI()

# ----------------------------------------------------
# 2. Define the LLM-Based Policy Function
# ----------------------------------------------------
def llm_policy(state):
    """
    Given the current state (a NumPy array of 4 floats for CartPole),
    this function sends a prompt to the LLM (e.g. GPT-4) to decide which action to take.
    
    The prompt instructs the model:
      - The state is provided as a list.
      - Action 0 means "move left" and action 1 means "move right".
      - Respond ONLY with 0 or 1.
      
    Returns:
      An integer action (0 or 1).
    """
    # Prepare a prompt with a description and the current state.
    prompt = (
        "You are an expert reinforcement learning agent controlling a cartpole. "
        "The environment state is given as a list of 4 numbers. "
        f"The current state is: {state.tolist()}. "
        "Action 0 corresponds to moving the cart left and action 1 corresponds to moving it right. "
        "Based on the state, decide the best action to take. "
        "Please respond with ONLY a single number: 0 or 1."
    )

    try:
        response = client.chat.completions(
            model="gpt-4",  
            messages=[
                {"role": "system", "content": "You are an expert RL agent controlling a cartpole."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Lower temperature for deterministic output.
        )
        answer = response["choices"][0]["message"]["content"].strip()
        action = int(answer)
        if action not in [0, 1]:
            print(f"Unexpected action from LLM: {action}. Defaulting to 0.")
            action = 0
    except Exception as e:
        print(f"LLM API error: {e}. Defaulting action to 0.")
        action = 0

    # (Optional) Sleep briefly to avoid rate limits.
    time.sleep(0.5)
    return action

# ----------------------------------------------------
# 3. Evaluate One Episode Using the LLM Policy
# ----------------------------------------------------
def evaluate_policy_llm(render=False):
    """
    Runs one episode of the CartPole environment, using the LLM to choose actions.
    Returns the total reward for the episode.
    """
    obs, info = env.reset()
    total_reward = 0
    while True:
        if render:
            env.render()

        # Use the LLM to decide an action based on the current state.
        action = llm_policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if done or truncated:
            break
    return total_reward

# ----------------------------------------------------
# 4. Main: Run Several Episodes
# ----------------------------------------------------
if __name__ == "__main__":
    test_episodes = 5
    rewards = []
    
    for ep in range(test_episodes):
        R = evaluate_policy_llm(render=False)
        rewards.append(R)
        print(f"Episode {ep+1}: Total Reward = {R}")
    
    env.close()

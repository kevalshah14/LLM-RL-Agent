import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt  # <-- for plotting
from google import genai

# -------------
# LLM IMPORTS
# -------------
from openai import OpenAI
import json  # used for an optional JSON parse attempt

API_KEY = os.getenv('apiKey')
BASE_URL = os.getenv('baseURL')
GEMINI_API_KEY = os.getenv('geminiApiKey')

# --------------------------------
#  MemoryTable for Rewards
# --------------------------------
class MemoryTable:
    """
    A simple table storing (state, action, reward).
    We'll pass these to the LLM so it can figure out
    how to update the policy (instead of Q-values).
    """
    def __init__(self):
        self.table = []

    def to_string(self):
        """
        Convert to lines: "state | action | reward"
        where state is a 4D tuple of discrete bins.
        """
        header = "state (pos,vel,angle,angvel) | action | reward\n"
        header += "--------------------------------------------\n"
        lines = []
        for (s, a, r) in self.table:
            lines.append(f"{s} | {a} | {r}")
        return header + "\n".join(lines)


# --------------------------------
#  PolicyTable
# --------------------------------
class PolicyTable:
    """
    Stores a mapping from discrete states -> best action.
    We'll ask the LLM to produce lines like:
      pos_bin | vel_bin | angle_bin | angvel_bin | action
    and parse them here.
    """
    def __init__(self):
        self.table = []  # list of (state_tuple, action)

    def to_string(self):
        """
        Convert the policy table to lines:
        "pos_bin | vel_bin | angle_bin | angvel_bin | action"
        """
        header = "pos_bin | vel_bin | angle_bin | angvel_bin | action\n"
        header += "-------------------------------------------\n"
        lines = []
        for (s, a) in self.table:
            (p, v, ang, angv) = s
            lines.append(f"{p} | {v} | {ang} | {angv} | {a}")
        return header + "\n".join(lines)

    def get_dict(self):
        """
        Convert to a dict: dict[state_tuple] = action
        """
        d = {}
        for (s, a) in self.table:
            d[s] = a
        return d


# --------------------------------
#  LLMBrain
# --------------------------------
class LLMBrain:
    def __init__(self,
                 num_buckets=(10, 10, 20, 20),
                 env_action_space_n=2):
        # self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Where we store immediate rewards observed each episode
        self.reward_table = MemoryTable()

        # A policy table (state -> action)
        self.policy_table = PolicyTable()

        # LLM conversation context
        self.llm_conversation = []

        # Discretization settings for CartPole
        self.num_buckets = num_buckets
        self.state_bounds = [
            (-4.8, 4.8),     # cart position
            (-3.0, 3.0),     # cart velocity
            (-0.418, 0.418), # pole angle (~±24 degrees)
            (-4.0, 4.0)      # pole angular velocity
        ]

        # Number of actions in CartPole
        self.action_space_n = env_action_space_n

        # Store a history of reward tables (as strings) from past episodes.
        self.reward_history = []

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        """
        role: "user", "assistant", "system"
        """
        self.llm_conversation.append({"role": role, "content": text})

    def discretize_state(self, obs):
        """
        Convert a continuous CartPole state to a 4D discrete state.
        """
        ratios = []
        for i, val in enumerate(obs):
            low, high = self.state_bounds[i]
            clipped = max(min(val, high), low)
            ratio = (clipped - low) / (high - low)
            ratios.append(ratio)

        discrete_indices = []
        for i, ratio in enumerate(ratios):
            nb = self.num_buckets[i]
            idx = int(round((nb - 1) * ratio))
            discrete_indices.append(idx)

        return tuple(discrete_indices)

    def get_action(self, state, env):
        """
        Return the policy's action for the given state (discretized).
        If we don't know, pick random.
        """
        disc_state = self.discretize_state(state)
        policy_dict = self.policy_table.get_dict()
        if disc_state in policy_dict:
            return policy_dict[disc_state]
        else:
            return env.action_space.sample()

    def add_to_reward_table(self, state, action, reward):
        """
        Store the immediate reward for (state, action).
        """
        disc_state = self.discretize_state(state)
        self.reward_table.table.append((disc_state, action, reward))

    def query_llm(self):
        """
        Send the conversation to the LLM and get the response text.
        Added debug prints to show the raw response.
        """
        prompt = " ".join(msg.get("content", "") for msg in self.llm_conversation)
        model = "gemini-2.0-flash"
        
        for attempt in range(5):
            try:
                print(f"Attempting with model: {model}")
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                # The API returns the generated text in the 'text' attribute.
                text_response = response.text

                # DEBUG: Print the raw LLM response
                print("LLM raw response:")
                print(text_response)
                
                # Optionally, add the assistant's response to the conversation history.
                self.add_llm_conversation(text_response, "assistant")
                return text_response
            
            except Exception as e:
                print(f"Error with model {model}: {e}")
                if attempt < 4:
                    print("Retrying...")
                    print("Waiting for 120 seconds before retrying...")
                    time.sleep(120)
                else:
                    print(f"Failed with model: {model} after 5 attempts")
                    break  # Exit the loop after 5 failed attempts
        
        raise Exception(f"{model} failed after 5 attempts.")

    def llm_update_policy(self):
        """
        Provide the reward table history (last 10 episodes) + existing policy to the LLM,
        ask for an updated policy. Then parse it.
        """
        self.reset_llm_conversation()

        # Concatenate the last 10 reward tables (if available)
        reward_history_str = "\n\n".join(self.reward_history[-10:])

        # Build a system/user prompt
        system_prompt = f"""
You are an AI that decides a simple tabular policy for CartPole.
The environment has 4 discrete dimensions (pos_bin, vel_bin, angle_bin, angvel_bin)
and 2 possible actions (0 or 1).

Generate and run code for the calculation of the new policy based on the reward table history and the old policy.

We have a history of immediate rewards from the last 10 episodes (state, action, reward).
We also have the old policy table (mapping state->action).
We want you to propose a new policy.

Format the new policy as lines:
  pos_bin | vel_bin | angle_bin | angvel_bin | action

where "action" is either 0 or 1.
Only output these lines (no explanations).
        """

        old_policy_str = self.policy_table.to_string()

        user_prompt = f"""
---------------------
Reward Table History (last 10 episodes):
{reward_history_str}

---------------------
Old Policy Table:
{old_policy_str}

Please output ONLY the new policy in lines of:
pos_bin | vel_bin | angle_bin | angvel_bin | action
        """

        self.add_llm_conversation(system_prompt, "system")
        self.add_llm_conversation(user_prompt, "user")

        # Query the LLM
        new_policy_str = self.query_llm()

        # DEBUG: Print the LLM response before attempting to parse it
        print("LLM Response:")
        print(new_policy_str)

        # Attempt to parse JSON if the LLM returned JSON (not expected but for debugging)
        try:
            policy_json = json.loads(new_policy_str)
            print("Successfully parsed JSON from LLM response.")
        except json.JSONDecodeError:
            print("LLM response is not valid JSON. Proceeding with text line parsing.")

        # Now parse `new_policy_str`
        # Clear old policy
        self.policy_table.table = []
        parsed_count = 0

        for line in new_policy_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 5:
                try:
                    p_bin = int(parts[0].strip())
                    v_bin = int(parts[1].strip())
                    ang_bin = int(parts[2].strip())
                    angv_bin = int(parts[3].strip())
                    act = int(parts[4].strip())
                    state = (p_bin, v_bin, ang_bin, angv_bin)
                    self.policy_table.table.append((state, act))
                    parsed_count += 1
                except ValueError:
                    print(f"Failed to parse line: {line}")
        print(f"Successfully parsed {parsed_count} policy entries from LLM response.")


# --------------------------------
#  MAIN LOOP
# --------------------------------
def main():
    env = gym.make('CartPole-v1', render_mode=None)
    llm_brain = LLMBrain(num_buckets=(10, 10, 20, 20),
                         env_action_space_n=env.action_space.n)

    folder = './cartpole_llm_no_q_logs/'
    os.makedirs(folder, exist_ok=True)

    NUM_EPISODES = 150
    
    # -- Lists to store rewards for plotting --
    training_rewards = []
    test_avg_rewards = []  # We'll store the average reward across 5 tests
    
    for episode in range(NUM_EPISODES):
        # Clear old reward table for each new episode
        llm_brain.reward_table.table = []

        state, _ = env.reset()
        done = False
        step_id = 0
        total_reward = 0

        # Logging
        episode_folder = os.path.join(folder, f"episode_{episode+1}")
        os.makedirs(episode_folder, exist_ok=True)
        train_file = os.path.join(episode_folder, "training_episode.txt")

        with open(train_file, 'w') as f:
            f.write(f"Episode {episode+1}\n")
            f.write("Step | State (continuous) | Discretized State | Action | Reward\n")

            while not done:
                step_id += 1
                action = llm_brain.get_action(state, env)
                next_state, reward, done, info, _ = env.step(action)

                total_reward += reward
                llm_brain.add_to_reward_table(state, action, reward)

                disc_st = llm_brain.discretize_state(state)
                f.write(f"{step_id} | {state} | {disc_st} | {action} | {reward}\n")

                state = next_state
                if done or step_id >= 200:
                    print(f"[Train] Ep {episode+1} ended. Total reward: {total_reward}")
                    break
            f.write(f"Total Reward: {total_reward}\n")
        
        # Track the training reward for plotting
        training_rewards.append(total_reward)
        
        # Save the current episode's reward table into the history (keeping only the last 10)
        reward_snapshot = llm_brain.reward_table.to_string()
        llm_brain.reward_history.append(reward_snapshot)
        if len(llm_brain.reward_history) > 10:
            llm_brain.reward_history.pop(0)

        # Now ask LLM to provide a new policy using the reward history
        llm_brain.llm_update_policy()

        # Save the new policy to file
        policy_file = os.path.join(episode_folder, "policy_table.txt")
        with open(policy_file, 'w') as f:
            f.write(llm_brain.policy_table.to_string())

        # Test the new policy
        TEST_EPISODES = 5
        test_rewards_this_update = []
        for test_i in range(TEST_EPISODES):
            state, _ = env.reset()
            done = False
            step_id = 0
            total_test_reward = 0

            test_file = os.path.join(episode_folder, f"testing_episode_{test_i+1}.txt")
            with open(test_file, 'w') as f:
                f.write(f"Testing Episode {test_i+1}\n")
                f.write("Step | State (continuous) | Discretized State | Action | Reward\n")

                while not done:
                    step_id += 1
                    action = llm_brain.get_action(state, env)
                    next_state, reward, done, info, _ = env.step(action)

                    total_test_reward += reward
                    disc_st = llm_brain.discretize_state(state)
                    f.write(f"{step_id} | {state} | {disc_st} | {action} | {reward}\n")

                    state = next_state
                    if done or step_id >= 200:
                        print(f"[Test] Ep {episode+1}, Test {test_i+1} ended. R: {total_test_reward}")
                        break
                f.write(f"Total Reward: {total_test_reward}\n")

            test_rewards_this_update.append(total_test_reward)

        # Store the average test reward for plotting
        avg_test_reward = np.mean(test_rewards_this_update)
        test_avg_rewards.append(avg_test_reward)

    env.close()

    # -------------------
    #  Plotting Results
    # -------------------
    # Plot training episode rewards
    plt.figure(figsize=(8, 4))
    plt.plot(training_rewards, label='Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Episode Rewards')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot average test rewards (post-update each episode)
    plt.figure(figsize=(8, 4))
    plt.plot(test_avg_rewards, label='Avg Test Rewards', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (5 tests)')
    plt.title('Average Test Rewards (every LLM policy update)')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

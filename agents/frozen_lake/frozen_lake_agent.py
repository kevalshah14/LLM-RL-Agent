import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt  # for plotting
from google import genai
import copy

# -------------
# LLM IMPORTS
# -------------
from openai import OpenAI
import json  # for JSON parsing/formatting
import csv
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('apiKey')
BASE_URL = os.getenv('baseURL')
GEMINI_API_KEY = os.getenv('geminiApiKey')


# --------------------------------
#  MemoryTable for Rewards (FrozenLake)
# --------------------------------
class MemoryTable:
    """
    A simple table storing (state, action, reward).
    For FrozenLake, the state is a single discrete integer.
    """
    def __init__(self):
        self.table = []

    def to_string(self):
        """
        Convert to lines: "state | action | reward"
        """
        header = "state | action | reward\n"
        header += "-----------------------\n"
        lines = []
        for (s, a, r) in self.table:
            lines.append(f"{s} | {a} | {r}")
        return header + "\n".join(lines)

    def to_json(self, total_reward=None, final_step=None):
        """
        Convert the reward table to a JSON-formatted string.
        Each entry is a dict with keys: state, action, reward.
        Optionally include the total_reward and the final_step (the step at which the episode terminated).
        """
        rewards_list = []
        for (s, a, r) in self.table:
            rewards_list.append({
                "state": int(s),
                "action": int(a),
                "reward": float(r) if isinstance(r, np.floating) else int(r) if isinstance(r, np.integer) else r
            })
        data = {"rewards": rewards_list}
        if total_reward is not None:
            data["total_reward"] = float(total_reward) if isinstance(total_reward, np.floating) else int(total_reward) if isinstance(total_reward, np.integer) else total_reward
        if final_step is not None:
            data["final_step"] = final_step
        return json.dumps(data, indent=2)


# --------------------------------
#  PolicyTable for FrozenLake
# --------------------------------
class PolicyTable:
    """
    Stores a mapping from state -> best action.
    The LLM is expected to output lines like:
      state | action
    """
    def __init__(self):
        self.table = []  # list of (state, action)

    def to_string(self):
        """
        Convert the policy table to lines:
        "state | action"
        """
        header = "state | action\n"
        header += "---------------\n"
        lines = []
        for (s, a) in self.table:
            lines.append(f"{s} | {a}")
        return header + "\n".join(lines)

    def get_dict(self):
        """
        Convert to a dict: dict[state] = action
        """
        d = {}
        for (s, a) in self.table:
            d[s] = a
        return d


# --------------------------------
#  LLMBrain for FrozenLake
# --------------------------------
class LLMBrain:
    def __init__(self, env_action_space_n=4):
        # Initialize the LLM client.
        # self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Where we store immediate rewards observed in each episode
        self.reward_table = MemoryTable()

        # A policy table (state -> action)
        self.policy_table = PolicyTable()
        
        # Store the BEST policy seen so far to prevent catastrophic forgetting
        self.best_policy_table = PolicyTable()
        self.best_avg_test_reward = -float('inf')

        # LLM conversation context
        self.llm_conversation = []

        # Number of actions in FrozenLake (usually 4: Left, Down, Right, Up)
        self.action_space_n = env_action_space_n

        # Store a history of reward tables.
        # Each entry is a tuple: (total_reward, reward_table_json_string)
        self.reward_history = []

        # We will later set state-to-coordinate mapping here.
        self.state_coords_mapping = {}

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        """
        role: "user", "assistant", "system"
        """
        self.llm_conversation.append({"role": role, "content": text})

    def discretize_state(self, obs):
        """
        For FrozenLake, the state is already discrete.
        If needed, cast the observation to an integer.
        """
        try:
            return int(obs)
        except Exception:
            return obs

    def get_action(self, state, env, use_best=False):
        """
        Return the policy's action for the given state.
        If use_best is True, uses the stored best policy.
        If we don't have a policy for the state, choose a random action.
        """
        disc_state = self.discretize_state(state)
        
        if use_best:
            policy_dict = self.best_policy_table.get_dict()
        else:
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
        """
        prompt = " ".join(msg.get("content", "") for msg in self.llm_conversation)
        model = "gemini-2.5-flash"
        
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
                    print("Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"Failed with model: {model} after 5 attempts")
                    break
        
        raise Exception(f"{model} failed after 5 attempts.")

    def llm_update_policy(self):
        """
        Provide the reward table history (from the best 5 episodes) along with the existing policy,
        the state-to-coordinate mapping, and extra descriptive context to the LLM so that it better understands the dynamics.
        Then, ensure that a complete policy (covering every state) is returned.
        The reward table now includes a 'final_step' field to indicate the step at which the episode terminated.
        """
        self.reset_llm_conversation()

        # -----------------------------
        # Select the best 5 reward histories based on total reward.
        # -----------------------------
        if self.reward_history:
            sorted_reward_history = sorted(self.reward_history, key=lambda x: x[0], reverse=True)
            best_reward_history = sorted_reward_history[:5]
            reward_history_json = json.dumps([json.loads(entry[1]) for entry in best_reward_history], indent=2)
        else:
            reward_history_json = "[]"
        
        # Prepare the state-to-coordinate mapping as JSON.
        state_coords_mapping_str = json.dumps(self.state_coords_mapping, indent=2)
        n_states = len(self.state_coords_mapping)

        # Build system and user prompts with additional context.
        system_prompt = f"""
You are an AI that decides a simple tabular policy for the FrozenLake environment.
Here are some important details about FrozenLake:
- The environment is represented as a grid. Each state is a number that maps to (row, col) coordinates.
- There are 4 possible actions: 0 (Left), 1 (Down), 2 (Right), 3 (Up).
- The objective is to navigate from the start state to the goal state.
- In this environment:
    • Stepping onto a frozen surface gives 0 reward.
    • Falling into a hole results in a reward of 0 and ends the episode.
    • Reaching the goal gives a reward of 1.
- The grid mapping for the states is as follows:
{state_coords_mapping_str}
        """

        old_policy_str = self.policy_table.to_string()

        # The user prompt now instructs the LLM to output a complete policy covering every state.
        # It also mentions that each reward history includes a 'final_step' indicating the step at which the episode terminated.
        user_prompt = f"""
Below is the reward history (best 5 episodes) in JSON format.
Each reward entry contains:
- state: the state number,
- action: the action taken,
- reward: the immediate reward received,
- final_step: the step at which the episode terminated (indicating where it failed or ended).

Reward History:
{reward_history_json}

Here is the old policy table for reference:
{old_policy_str}

There are {n_states} states in the FrozenLake environment (numbered from 0 to {n_states - 1}).
Please analyze the reward outcomes in the context of the FrozenLake environment and output ONLY the new **complete policy**.
For each state from 0 to {n_states - 1}, provide the best action to take (i.e. the action that maximizes the chance to reach the goal while avoiding holes) in the following format on a separate line:
state | action

Ensure that your answer includes an entry for every state.
        """

        # Add messages to the conversation
        self.add_llm_conversation(system_prompt, "system")
        self.add_llm_conversation(user_prompt, "user")

        # Query the LLM
        new_policy_str = self.query_llm()

        # DEBUG: Print the LLM response before parsing
        print("LLM Response:")
        print(new_policy_str)

        # Attempt to parse JSON if the LLM returned JSON (optional)
        try:
            policy_json = json.loads(new_policy_str)
            print("Successfully parsed JSON from LLM response.")
        except json.JSONDecodeError:
            print("LLM response is not valid JSON. Proceeding with text line parsing.")

        # Now parse new_policy_str (expected format: "state | action")
        # Clear the existing policy table.
        self.policy_table.table = []
        parsed_count = 0

        for line in new_policy_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Skip header lines if any
            if "state" in line.lower() and "action" in line.lower():
                continue
            parts = line.split("|")
            if len(parts) == 2:
                try:
                    state_val = int(parts[0].strip())
                    act = int(parts[1].strip())
                    self.policy_table.table.append((state_val, act))
                    parsed_count += 1
                except ValueError:
                    print(f"Failed to parse line: {line}")
        print(f"Successfully parsed {parsed_count} policy entries from LLM response.")

        # -----------------------------
        # Ensure that the policy covers every state
        # -----------------------------
        policy_dict = self.policy_table.get_dict()
        for s in range(n_states):
            if s not in policy_dict:
                default_action = 0  # You can choose a different default action if desired.
                print(f"State {s} missing in new policy. Filling with default action {default_action}.")
                policy_dict[s] = default_action

        # Rebuild the full policy table from the complete policy dictionary (sorted by state)
        self.policy_table.table = sorted(policy_dict.items(), key=lambda x: x[0])
        print("Updated complete policy:")
        print(self.policy_table.to_string())


# --------------------------------
#  Helper function to get coordinates for a state
# --------------------------------
def get_state_coordinates(env, state):
    """
    Convert the state (an integer) to (row, col) coordinates using the environment's grid.
    """
    grid = env.unwrapped.desc
    ncol = grid.shape[1]
    row = state // ncol
    col = state % ncol
    return (row, col)


# --------------------------------
#  Helper function for smoothing data
# --------------------------------
def smooth_data(data, window=5):
    """
    Smooth the data using a simple moving average with the given window size.
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')


# --------------------------------
#  MAIN LOOP for FrozenLake with Coordinates Logging and Cycle Detection
# --------------------------------
def main():
    # Create the FrozenLake environment.
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="human")
    llm_brain = LLMBrain(env_action_space_n=env.action_space.n)

    # Build a complete state-to-coordinate mapping for the grid and store it in llm_brain.
    grid = env.unwrapped.desc
    nrow, ncol = grid.shape
    state_coords_mapping = {}
    for s in range(nrow * ncol):
        state_coords_mapping[s] = (s // ncol, s % ncol)
    llm_brain.state_coords_mapping = state_coords_mapping

    folder = './logs/frozen_lake/'
    os.makedirs(folder, exist_ok=True)

    NUM_EPISODES = 100
    training_rewards = []
    test_avg_rewards = []  # Average reward across test episodes after each policy update

    # Maximum allowed visits to a state before triggering a random action (cycle detection)
    max_visits = 3

    for episode in range(NUM_EPISODES):
        # Clear the reward table for the new episode
        llm_brain.reward_table.table = []

        state, _ = env.reset()
        done = False
        step_id = 0
        total_reward = 0

        # Dictionary to track number of visits per state (for cycle detection)
        visited_states = {}

        episode_folder = os.path.join(folder, f"episode_{episode+1}")
        os.makedirs(episode_folder, exist_ok=True)
        train_file = os.path.join(episode_folder, "training_episode.txt")

        with open(train_file, 'w') as f:
            f.write(f"Episode {episode+1}\n")
            f.write("Step | State (Coordinates) | Action | Next State (Coordinates) | Env Reward\n")

            while not done:
                step_id += 1

                # Cycle detection: update visited count and check threshold
                visited_states[state] = visited_states.get(state, 0) + 1
                if visited_states[state] > max_visits:
                    print(f"[Train] State {state} visited {visited_states[state]} times. Taking a random action to break cycle.")
                    action = env.action_space.sample()
                else:
                    action = llm_brain.get_action(state, env)

                # For gymnasium, step returns: observation, reward, terminated, truncated, info
                next_state, env_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Get coordinates for the current and next states.
                state_coords = get_state_coordinates(env, state)
                next_state_coords = get_state_coordinates(env, next_state)

                total_reward += env_reward  # Using the raw environment reward

                # Log to the reward table (using raw env_reward)
                llm_brain.add_to_reward_table(state, action, env_reward)

                f.write(f"{step_id} | {state} {state_coords} | {action} | {next_state} {next_state_coords} | {env_reward}\n")

                state = next_state

                if done or step_id >= 100:  # FrozenLake episodes are usually short
                    print(f"[Train] Episode {episode+1} ended. Total raw reward: {total_reward} at step {step_id}")
                    break
            f.write(f"Total Raw Reward: {total_reward}\n")
        
        training_rewards.append(total_reward)
        
        # Save the current episode's reward table along with its total reward and the final (failure) step.
        reward_snapshot = llm_brain.reward_table.to_json(total_reward, final_step=step_id)
        llm_brain.reward_history.append((total_reward, reward_snapshot))

        # Update the policy using the LLM
        llm_brain.llm_update_policy()

        # Test the updated policy using the same coordinate logging and cycle detection
        TEST_EPISODES = min(3 + episode // 20, 5)
        test_rewards_this_update = []
        for test_i in range(TEST_EPISODES):
            state, _ = env.reset()
            done = False
            step_id = 0
            total_test_reward = 0

            # Track visited states during testing to prevent cycles
            visited_states_test = {}

            test_file = os.path.join(episode_folder, f"testing_episode_{test_i+1}.txt")
            with open(test_file, 'w') as f:
                f.write(f"Testing Episode {test_i+1}\n")
                f.write("Step | State (Coordinates) | Action | Next State (Coordinates) | Env Reward\n")

                while not done:
                    step_id += 1

                    # Cycle detection for test episode
                    visited_states_test[state] = visited_states_test.get(state, 0) + 1
                    if visited_states_test[state] > max_visits:
                        print(f"[Test] State {state} visited {visited_states_test[state]} times. Taking a random action to break cycle.")
                        action = env.action_space.sample()
                    else:
                        action = llm_brain.get_action(state, env)

                    next_state, env_reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    state_coords = get_state_coordinates(env, state)
                    next_state_coords = get_state_coordinates(env, next_state)
                    total_test_reward += env_reward

                    f.write(f"{step_id} | {state} {state_coords} | {action} | {next_state} {next_state_coords} | {env_reward}\n")

                    state = next_state

                    if done or step_id >= 100:
                        print(f"[Test] Episode {episode+1}, Test {test_i+1} ended. Total raw reward: {total_test_reward} at step {step_id}")
                        break
                f.write(f"Total Raw Reward: {total_test_reward}\n")

            test_rewards_this_update.append(total_test_reward)

        # Store the average test reward for plotting
        avg_test_reward = np.mean(test_rewards_this_update)
        test_avg_rewards.append(avg_test_reward)
        
        print(f"Current Policy Avg Reward: {avg_test_reward:.2f} | Best So Far: {llm_brain.best_avg_test_reward:.2f}")

        # Policy Update & Retention Logic
        if avg_test_reward > llm_brain.best_avg_test_reward:
            print(f"New Best Policy Found! Updating Best Policy (Reward: {avg_test_reward:.2f})")
            llm_brain.best_avg_test_reward = avg_test_reward
            llm_brain.best_policy_table = copy.deepcopy(llm_brain.policy_table)
        else:
            print(f"Current policy ({avg_test_reward:.2f}) is worse than best ({llm_brain.best_avg_test_reward:.2f}). Reverting to Best Policy.")
            llm_brain.policy_table = copy.deepcopy(llm_brain.best_policy_table)
        
        # Save the policy table to file
        policy_file = os.path.join(episode_folder, "policy_table.txt")
        with open(policy_file, 'w') as f:
            f.write(llm_brain.policy_table.to_string())

    env.close()

    # -------------------
    #  Save Plot Values and Figures
    # -------------------
    csv_filename = os.path.join(folder, "frozenlake_plot_values.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Training Raw Reward", "Avg Test Raw Reward"])
        for i in range(len(training_rewards)):
            writer.writerow([i+1, training_rewards[i], test_avg_rewards[i]])
    print(f"Saved plot values to {csv_filename}")

    # Plot training rewards with smoothing
    window_size = 5
    smoothed_training = smooth_data(training_rewards, window=window_size)
    training_x = np.arange(window_size - 1, len(training_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(training_rewards, label='Training Raw Rewards', alpha=0.3, marker='o')
    plt.plot(training_x, smoothed_training, label='Smoothed Trend', color='blue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Raw Reward')
    plt.title('Training Episode Raw Rewards for FrozenLake')
    plt.legend()
    plt.tight_layout()
    training_plot_filename = os.path.join(folder, "frozenlake_training_rewards_smoothed.png")
    plt.savefig(training_plot_filename)
    print(f"Saved training rewards plot to {training_plot_filename}")
    plt.show()

    # Plot average test rewards with smoothing
    smoothed_test = smooth_data(test_avg_rewards, window=window_size)
    test_x = np.arange(window_size - 1, len(test_avg_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(test_avg_rewards, label='Avg Test Raw Rewards', color='orange', alpha=0.3, marker='o')
    plt.plot(test_x, smoothed_test, label='Smoothed Trend', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Raw Reward (Test Episodes)')
    plt.title('Average Test Raw Rewards for FrozenLake')
    plt.legend()
    plt.tight_layout()
    test_plot_filename = os.path.join(folder, "frozenlake_avg_test_rewards_smoothed.png")
    plt.savefig(test_plot_filename)
    print(f"Saved average test rewards plot to {test_plot_filename}")
    plt.show()


if __name__ == "__main__":
    main()

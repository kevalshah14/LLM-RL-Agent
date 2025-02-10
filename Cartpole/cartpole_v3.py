import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt  

# ---------------------------
# LLM and Utility Imports
# ---------------------------
from openai import OpenAI
from google import genai
import json 
import csv  

# Retrieve API keys and base URLs from environment variables
API_KEY = os.getenv('apiKey')
BASE_URL = os.getenv('baseURL')
GEMINI_API_KEY = os.getenv('geminiApiKey')


# --------------------------------------
# MemoryTable: Stores (state, action, reward)
# --------------------------------------
class MemoryTable:
    """
    A simple container to store triplets of (state, action, reward).
    These entries are later sent to the LLM so it can learn and update the policy.
    """
    def __init__(self):
        self.table = []

    def to_string(self):
        """
        Return a string representation of the table with headers.
        Each line has the format:
            "state (pos,vel,angle,angvel) | action | reward"
        """
        header = "state (pos,vel,angle,angvel) | action | reward\n"
        header += "--------------------------------------------\n"
        lines = []
        for (s, a, r) in self.table:
            lines.append(f"{s} | {a} | {r}")
        return header + "\n".join(lines)

    def to_json(self, total_reward=None):
        """
        Convert the reward table into a JSON-formatted string.
        Each entry is a dictionary with keys: state, action, reward.
        Optionally includes the total_reward of an episode.
        """
        rewards_list = []
        for (s, a, r) in self.table:
            rewards_list.append({
                "state": [int(x) for x in s],
                "action": int(a),
                "reward": float(r) if isinstance(r, np.floating) else int(r) if isinstance(r, np.integer) else r
            })
        data = {"rewards": rewards_list}
        if total_reward is not None:
            data["total_reward"] = float(total_reward) if isinstance(total_reward, np.floating) else int(total_reward) if isinstance(total_reward, np.integer) else total_reward
        return json.dumps(data, indent=2)


# --------------------------------------
# PolicyTable: Maps discretized states to best action
# --------------------------------------
class PolicyTable:
    """
    Stores the learned mapping from discrete states to the optimal action.
    The expected format for each policy entry is:
        pos_bin | vel_bin | angle_bin | angvel_bin | action
    """
    def __init__(self):
        # Each entry is a tuple: (state_tuple, action)
        self.table = []

    def to_string(self):
        """
        Return a string representation of the policy table with headers.
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
        Convert the list of policy entries into a dictionary for easy lookup:
            dict[state_tuple] = action
        """
        d = {}
        for (s, a) in self.table:
            d[s] = a
        return d


# --------------------------------------
# LLMBrain: Uses an LLM to update and refine the policy
# --------------------------------------
class LLMBrain:
    def __init__(self,
                 num_buckets=(10, 10, 20, 20),
                 env_action_space_n=2):
        # Instantiate the LLM client
        # self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Memory table for storing immediate rewards during an episode
        self.reward_table = MemoryTable()

        # Table mapping discretized states to best actions
        self.policy_table = PolicyTable()

        # List to store the conversation history with the LLM
        self.llm_conversation = []

        # Discretization configuration for the CartPole state space.
        # Each state (cart position, cart velocity, pole angle, pole angular velocity)
        # will be discretized into a fixed number of buckets.
        self.num_buckets = num_buckets
        self.state_bounds = [
            (-4.8, 4.8),     # Cart position bounds
            (-3.0, 3.0),     # Cart velocity bounds
            (-0.418, 0.418), # Pole angle bounds (approximately ±24 degrees)
            (-4.0, 4.0)      # Pole angular velocity bounds
        ]

        # Number of actions available in CartPole (typically 2: left or right)
        self.action_space_n = env_action_space_n

        # History of reward tables from episodes.
        # Each entry is a tuple: (total_reward, reward_table_json_string)
        self.reward_history = []

    def reset_llm_conversation(self):
        """Reset the conversation history with the LLM."""
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        """
        Append a new message to the LLM conversation history.
        role: "user", "assistant", or "system"
        """
        self.llm_conversation.append({"role": role, "content": text})

    def discretize_state(self, obs):
        """
        Convert the continuous state from the CartPole environment into a discrete state.
        Each dimension is scaled to a value between 0 and (num_buckets-1) based on predefined bounds.
        
        Args:
            obs: The continuous state (list or tuple of 4 values).
            
        Returns:
            A tuple of discretized indices.
        """
        ratios = []
        for i, val in enumerate(obs):
            low, high = self.state_bounds[i]
            # Clip the value to lie within the specified bounds
            clipped = max(min(val, high), low)
            ratio = (clipped - low) / (high - low)
            ratios.append(ratio)

        discrete_indices = []
        for i, ratio in enumerate(ratios):
            nb = self.num_buckets[i]
            # Map the continuous ratio to a discrete bucket index
            idx = int(round((nb - 1) * ratio))
            discrete_indices.append(idx)

        return tuple(discrete_indices)

    def get_action(self, state, env):
        """
        Determine the action to take given the current state.
        Uses the learned policy if available; otherwise, selects a random action.
        
        Args:
            state: The current continuous state.
            env: The environment (used for random action sampling if needed).
            
        Returns:
            An integer representing the action.
        """
        disc_state = self.discretize_state(state)
        policy_dict = self.policy_table.get_dict()
        # Return the learned action if this state exists in the policy, else random action.
        if disc_state in policy_dict:
            return policy_dict[disc_state]
        else:
            return env.action_space.sample()

    def add_to_reward_table(self, state, action, reward):
        """
        Save the observed (discretized) state, action, and immediate reward to the reward table.
        """
        disc_state = self.discretize_state(state)
        self.reward_table.table.append((disc_state, action, reward))

    def query_llm(self):
        """
        Send the current conversation (context + prompt) to the LLM and return its response.
        Retries up to 5 times if an error occurs.
        """
        # Build the prompt by concatenating all messages from the conversation history.
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

                # DEBUG: Output the raw response from the LLM.
                print("LLM raw response:")
                print(text_response)
                
                # Append the LLM's answer to the conversation history.
                self.add_llm_conversation(text_response, "assistant")
                return text_response
            
            except Exception as e:
                print(f"Error with model {model}: {e}")
                if attempt < 4:
                    print("Retrying...")
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    print(f"Failed with model: {model} after 5 attempts")
                    break  # Exit after 5 failed attempts
        
        raise Exception(f"{model} failed after 5 attempts.")

    def llm_update_policy(self):
        """
        Use the LLM to generate an updated policy based on the best reward histories.
        
        The process involves:
          1. Selecting the best 5 episodes (by total reward) from the reward history.
          2. Creating a prompt that includes these reward tables (in JSON format)
             and the current policy table.
          3. Sending this prompt to the LLM and parsing its output.
          4. Updating the local policy table with the newly generated policy.
        """
        # Reset conversation history before constructing a new prompt.
        self.reset_llm_conversation()

        # -----------------------------
        # Select the best 5 episodes from reward_history
        # -----------------------------
        if self.reward_history:
            # Sort reward history by total reward in descending order.
            sorted_reward_history = sorted(self.reward_history, key=lambda x: x[0], reverse=True)
            best_reward_history = sorted_reward_history[:5]
            # Convert each reward table JSON string back into a dict and then dump as a JSON array.
            reward_history_json = json.dumps([json.loads(entry[1]) for entry in best_reward_history], indent=2)
        else:
            reward_history_json = "[]"
        
        # -----------------------------
        # Build system and user prompts for the LLM
        # -----------------------------
        system_prompt = f"""
        You are an AI that decides a simple tabular policy for CartPole.
        The environment has 4 discrete dimensions (pos_bin, vel_bin, angle_bin, angvel_bin)
        and 2 possible actions (0 or 1).

        Note: The reward tables are provided in JSON format for easy parsing.
        """

        # Get the current policy as a formatted string.
        old_policy_str = self.policy_table.to_string()

        user_prompt = f"""
        ---------------------
        Reward Table History (best 5 episodes) in JSON format:
        {reward_history_json}

        ---------------------
        Old Policy Table:
        {old_policy_str}

        Please output ONLY the new policy in lines of:
        pos_bin | vel_bin | angle_bin | angvel_bin | action
        """

        # Add the system and user messages to the conversation.
        self.add_llm_conversation(system_prompt, "system")
        self.add_llm_conversation(user_prompt, "user")

        # Send the prompt to the LLM and obtain its response.
        new_policy_str = self.query_llm()

        # DEBUG: Print the LLM response for debugging purposes.
        print("LLM Response:")
        print(new_policy_str)

        # Attempt to parse the LLM response as JSON (if applicable)
        try:
            policy_json = json.loads(new_policy_str)
            print("Successfully parsed JSON from LLM response.")
        except json.JSONDecodeError:
            print("LLM response is not valid JSON. Proceeding with text line parsing.")

        # -----------------------------
        # Parse the LLM response to update the policy table
        # -----------------------------
        # Clear the old policy entries
        self.policy_table.table = []
        parsed_count = 0

        # Expecting one policy entry per line, with each line having 5 fields separated by '|'
        for line in new_policy_str.split("\n"):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.split("|")
            if len(parts) == 5:
                try:
                    # Convert each part to an integer after stripping whitespace.
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


# --------------------------------------
# Helper Function: Data Smoothing for Plotting
# --------------------------------------
def smooth_data(data, window=5):
    """
    Smooth the input data using a simple moving average.
    
    Args:
        data (list or array): Data to smooth.
        window (int): The window size for the moving average.
    
    Returns:
        The smoothed data as a NumPy array.
    """
    return np.convolve(data, np.ones(window) / window, mode='valid')


# --------------------------------------
# Main Loop: Train and Test Policy using LLM Updates
# --------------------------------------
def main():
    # Initialize the CartPole environment without rendering.
    env = gym.make('CartPole-v1', render_mode=None)
    
    # Create an instance of the LLMBrain with discretization settings.
    llm_brain = LLMBrain(num_buckets=(10, 10, 20, 20),
                         env_action_space_n=env.action_space.n)

    # Folder to store logs and output files.
    folder = './cartpole_llm_no_q_logs/'
    os.makedirs(folder, exist_ok=True)

    NUM_EPISODES = 100  # Total number of training episodes
    
    # Lists to store rewards for later plotting.
    training_rewards = []    # Total reward per training episode.
    test_avg_rewards = []    # Average reward over 5 test episodes after each training episode.
    
    for episode in range(NUM_EPISODES):
        # Clear the reward table at the start of each training episode.
        llm_brain.reward_table.table = []

        # Reset the environment to start a new episode.
        state, _ = env.reset()
        done = False
        step_id = 0
        total_reward = 0

        # Create a folder for the current episode's logs.
        episode_folder = os.path.join(folder, f"episode_{episode+1}")
        os.makedirs(episode_folder, exist_ok=True)
        train_file = os.path.join(episode_folder, "training_episode.txt")

        # Open a file to log the training episode details.
        with open(train_file, 'w') as f:
            f.write(f"Episode {episode+1}\n")
            f.write("Step | State (continuous) | Discretized State | Action | Reward\n")

            # Run the training episode until termination or max steps reached.
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
        
        # Store training reward for this episode.
        training_rewards.append(total_reward)
        
        # Save the current episode's reward table as a JSON snapshot.
        reward_snapshot = llm_brain.reward_table.to_json(total_reward)
        llm_brain.reward_history.append((total_reward, reward_snapshot))

        # Update the policy using the LLM with the best 5 reward histories.
        llm_brain.llm_update_policy()

        # Save the new policy table to file.
        policy_file = os.path.join(episode_folder, "policy_table.txt")
        with open(policy_file, 'w') as f:
            f.write(llm_brain.policy_table.to_string())

        # -------------------------------
        # Test the new policy over a few episodes.
        # -------------------------------
        TEST_EPISODES = 5
        test_rewards_this_update = []
        for test_i in range(TEST_EPISODES):
            state, _ = env.reset()
            done = False
            step_id = 0
            total_test_reward = 0

            # Log details for each test episode.
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

        # Calculate and record the average reward across the test episodes.
        avg_test_reward = np.mean(test_rewards_this_update)
        test_avg_rewards.append(avg_test_reward)

    # Close the environment once all episodes are complete.
    env.close()

    # -------------------
    # Save training and test reward values to CSV for later analysis.
    # -------------------
    csv_filename = "plot_values.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Training Reward", "Avg Test Reward"])
        for i in range(len(training_rewards)):
            writer.writerow([i+1, training_rewards[i], test_avg_rewards[i]])
    print(f"Saved plot values to {csv_filename}")

    # -------------------
    # Plotting: Training and Test Rewards with Smoothing
    # -------------------
    # Define a window size for the moving average.
    window_size = 5

    # Smooth the training rewards.
    smoothed_training = smooth_data(training_rewards, window=window_size)
    # Adjust x-axis indices to match the smoothed data length.
    training_x = np.arange(window_size - 1, len(training_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(training_rewards, label='Training Rewards', alpha=0.3, marker='o')
    plt.plot(training_x, smoothed_training, label='Smoothed Trend', color='blue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Episode Rewards')
    plt.legend()
    plt.tight_layout()
    training_plot_filename = "training_rewards_smoothed.png"
    plt.savefig(training_plot_filename)
    print(f"Saved training rewards plot to {training_plot_filename}")
    plt.show()

    # Smooth the average test rewards.
    smoothed_test = smooth_data(test_avg_rewards, window=window_size)
    test_x = np.arange(window_size - 1, len(test_avg_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(test_avg_rewards, label='Avg Test Rewards', color='orange', alpha=0.3, marker='o')
    plt.plot(test_x, smoothed_test, label='Smoothed Trend', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (5 tests)')
    plt.title('Average Test Rewards (every LLM policy update)')
    plt.legend()
    plt.tight_layout()
    test_plot_filename = "avg_test_rewards_smoothed.png"
    plt.savefig(test_plot_filename)
    print(f"Saved average test rewards plot to {test_plot_filename}")
    plt.show()


# Entry point: Start the main training and testing loop.
if __name__ == "__main__":
    main()

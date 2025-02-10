import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from google import genai  # Assumes you have the proper gemini client installed

import json
import csv
import itertools
import random

# --------------------------------------------------
# Custom Reward Wrapper to Remove the -100 Penalty
# --------------------------------------------------
class RemovePenaltyWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    def reward(self, reward):
        # If the reward equals -100 (penalty), replace it with 0.
        # You can adjust the replacement value if desired.
        if reward == -100:
            return 0
        return reward

# --------------------------------------------------
# MemoryTable for Rewards
# --------------------------------------------------
class MemoryTable:
    """
    A simple table storing (state, action, reward).
    We'll pass these to the LLM so it can figure out
    how to update the policy.
    """
    def __init__(self):
        self.table = []

    def to_string(self):
        header = "state (angle_bin, angular_vel_bin, hor_speed_bin, ver_speed_bin) | action (vector) | reward\n"
        header += "--------------------------------------------------------------------------\n"
        lines = []
        for (s, a, r) in self.table:
            # Format the action vector nicely
            action_str = ", ".join([f"{x:.2f}" for x in a])
            lines.append(f"{s} | [{action_str}] | {r}")
        return header + "\n".join(lines)

    def to_json(self, total_reward=None):
        rewards_list = []
        for (s, a, r) in self.table:
            rewards_list.append({
                "state": [int(x) for x in s],
                "action": [float(x) for x in a],
                "reward": float(r) if isinstance(r, np.floating) else int(r) if isinstance(r, np.integer) else r
            })
        data = {"rewards": rewards_list}
        if total_reward is not None:
            data["total_reward"] = float(total_reward) if isinstance(total_reward, np.floating) else int(total_reward) if isinstance(total_reward, np.integer) else total_reward
        return json.dumps(data, indent=2)


# --------------------------------------------------
# PolicyTable
# --------------------------------------------------
class PolicyTable:
    """
    Stores a mapping from discretized states to the best continuous action.
    The expected policy table lines are:
      angle_bin | angular_vel_bin | hor_speed_bin | ver_speed_bin | action0, action1, action2, action3
    """
    def __init__(self):
        self.table = []  # list of (state_tuple, action_vector)

    def to_string(self):
        header = "angle_bin | angular_vel_bin | hor_speed_bin | ver_speed_bin | action (a0, a1, a2, a3)\n"
        header += "---------------------------------------------------------------------\n"
        lines = []
        for (s, a) in self.table:
            action_str = ", ".join([f"{x:.2f}" for x in a])
            (ang, ang_vel, hor_speed, ver_speed) = s
            lines.append(f"{ang} | {ang_vel} | {hor_speed} | {ver_speed} | {action_str}")
        return header + "\n".join(lines)

    def get_dict(self):
        d = {}
        for (s, a) in self.table:
            d[s] = a
        return d


# --------------------------------------------------
# LLMBrain
# --------------------------------------------------
class LLMBrain:
    def __init__(self,
                 num_buckets=(10, 10, 10, 10),
                 state_bounds=None,
                 discrete_actions=None):
        # Initialize the LLM client (using the Gemini API in this example)
        self.client = genai.Client(api_key=os.getenv('geminiApiKey'))

        # Memory for reward tables and policy
        self.reward_table = MemoryTable()
        self.policy_table = PolicyTable()
        self.llm_conversation = []

        # Use only 4 dimensions from the BipedalWalker observation:
        # For example, we use: hull angle, hull angular velocity,
        # horizontal speed, and vertical speed.
        self.num_buckets = num_buckets
        if state_bounds is None:
            # These bounds are approximate – adjust as needed!
            self.state_bounds = [(-1.0, 1.0), (-10.0, 10.0), (-3.0, 3.0), (-3.0, 3.0)]
        else:
            self.state_bounds = state_bounds

        # Create a discrete set of actions if not provided.
        # Here we use the Cartesian product of [-1.0, 0.0, 1.0] for each of the 4 action dimensions.
        if discrete_actions is None:
            self.discrete_actions = list(itertools.product([-1.0, 0.0, 1.0], repeat=4))
        else:
            self.discrete_actions = discrete_actions

        self.action_dim = 4
        self.reward_history = []  # List of tuples: (total_reward, reward_table_json_string)

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        self.llm_conversation.append({"role": role, "content": text})

    def discretize_state(self, obs):
        """
        Convert a continuous state (using the first 4 dimensions) to a discrete state.
        """
        # Use the first 4 features (assumed to be: hull angle, angular velocity, horizontal speed, vertical speed)
        relevant_obs = obs[:4]
        ratios = []
        for i, val in enumerate(relevant_obs):
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
        Returns the policy's action for the given state (discretized).
        If unknown, pick a random action from the discrete set.
        """
        disc_state = self.discretize_state(state)
        policy_dict = self.policy_table.get_dict()
        if disc_state in policy_dict:
            return policy_dict[disc_state]
        else:
            return random.choice(self.discrete_actions)

    def add_to_reward_table(self, state, action, reward):
        """
        Store the immediate reward for (state, action).
        """
        disc_state = self.discretize_state(state)
        self.reward_table.table.append((disc_state, action, reward))

    def query_llm(self):
        """
        Sends the conversation to the LLM and returns the response text.
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
                text_response = response.text
                print("LLM raw response:")
                print(text_response)
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
        Provide the reward table history and existing policy to the LLM,
        then parse its output to update the policy.
        """
        self.reset_llm_conversation()

        # Select the best 5 reward histories (if available)
        if self.reward_history:
            sorted_reward_history = sorted(self.reward_history, key=lambda x: x[0], reverse=True)
            best_reward_history = sorted_reward_history[:5]
            reward_history_json = json.dumps([json.loads(entry[1]) for entry in best_reward_history], indent=2)
        else:
            reward_history_json = "[]"
        
        system_prompt = f"""
You are an AI that decides a simple tabular policy for the BipedalWalker-v3 environment.
The state is discretized into 4 dimensions (angle_bin, angular_vel_bin, hor_speed_bin, ver_speed_bin).
The action is a 4-dimensional vector, where each element is one of -1.0, 0.0, or 1.0.
The reward tables are provided in JSON format.
        """

        old_policy_str = self.policy_table.to_string()
        user_prompt = f"""
---------------------
Reward Table History (best 5 episodes) in JSON format:
{reward_history_json}

---------------------
Old Policy Table:
{old_policy_str}

Please output ONLY the new policy in lines of:
angle_bin | angular_vel_bin | hor_speed_bin | ver_speed_bin | action0, action1, action2, action3
        """

        self.add_llm_conversation(system_prompt, "system")
        self.add_llm_conversation(user_prompt, "user")

        new_policy_str = self.query_llm()

        print("LLM Response:")
        print(new_policy_str)

        # Attempt JSON parse (if applicable)
        try:
            policy_json = json.loads(new_policy_str)
            print("Successfully parsed JSON from LLM response.")
        except json.JSONDecodeError:
            print("LLM response is not valid JSON. Proceeding with text line parsing.")

        # Clear the old policy table and parse the new policy
        self.policy_table.table = []
        parsed_count = 0

        for line in new_policy_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 5:
                try:
                    ang_bin = int(parts[0].strip())
                    ang_vel_bin = int(parts[1].strip())
                    hor_speed_bin = int(parts[2].strip())
                    ver_speed_bin = int(parts[3].strip())
                    action_str = parts[4].strip()
                    # Remove surrounding brackets if present
                    action_str = action_str.strip("[]")
                    action_parts = action_str.split(",")
                    if len(action_parts) != 4:
                        print(f"Failed to parse action vector in line: {line}")
                        continue
                    act = tuple(float(x.strip()) for x in action_parts)
                    state_tuple = (ang_bin, ang_vel_bin, hor_speed_bin, ver_speed_bin)
                    self.policy_table.table.append((state_tuple, act))
                    parsed_count += 1
                except ValueError:
                    print(f"Failed to parse line: {line}")
        print(f"Successfully parsed {parsed_count} policy entries from LLM response.")


# --------------------------------------------------
# Helper function for smoothing data
# --------------------------------------------------
def smooth_data(data, window=5):
    return np.convolve(data, np.ones(window) / window, mode='valid')


# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
def main():
    # Create the BipedalWalker-v3 environment
    env = gym.make('BipedalWalker-v3', hardcore=False, render_mode='human')
    # Wrap the environment to remove the -100 penalty from rewards
    env = RemovePenaltyWrapper(env)
    
    # Instantiate LLMBrain with appropriate discretization for BipedalWalker.
    llm_brain = LLMBrain(num_buckets=(10, 10, 10, 10))
    
    # Folder for logs
    folder = './bipedalwalker_llm_logs/'
    os.makedirs(folder, exist_ok=True)

    NUM_EPISODES = 100
    
    # Lists for plotting rewards
    training_rewards = []
    test_avg_rewards = []
    
    for episode in range(NUM_EPISODES):
        # Reset the reward table for this episode
        llm_brain.reward_table.table = []

        state, info = env.reset()
        done = False
        step_id = 0
        total_reward = 0

        # Create folder and file for logging this episode
        episode_folder = os.path.join(folder, f"episode_{episode+1}")
        os.makedirs(episode_folder, exist_ok=True)
        train_file = os.path.join(episode_folder, "training_episode.txt")

        with open(train_file, 'w') as f:
            f.write(f"Episode {episode+1}\n")
            f.write("Step | State (continuous) | Discretized State | Action | Reward\n")
            
            while not done:
                step_id += 1
                action = llm_brain.get_action(state, env)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                total_reward += reward
                llm_brain.add_to_reward_table(state, action, reward)

                disc_st = llm_brain.discretize_state(state)
                f.write(f"{step_id} | {state[:4]} | {disc_st} | {action} | {reward}\n")

                state = next_state
                if done or step_id >= 1600:
                    print(f"[Train] Episode {episode+1} ended. Total reward: {total_reward}")
                    break
            f.write(f"Total Reward: {total_reward}\n")
        
        training_rewards.append(total_reward)
        # Save the current reward table with its total reward (in JSON) for the LLM
        reward_snapshot = llm_brain.reward_table.to_json(total_reward)
        llm_brain.reward_history.append((total_reward, reward_snapshot))

        # Ask the LLM for an updated policy
        llm_brain.llm_update_policy()

        # Save the updated policy table to file
        policy_file = os.path.join(episode_folder, "policy_table.txt")
        with open(policy_file, 'w') as f:
            f.write(llm_brain.policy_table.to_string())

        # Testing phase with the new policy
        TEST_EPISODES = 5
        test_rewards_this_update = []
        for test_i in range(TEST_EPISODES):
            state, info = env.reset()
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
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    total_test_reward += reward
                    disc_st = llm_brain.discretize_state(state)
                    f.write(f"{step_id} | {state[:4]} | {disc_st} | {action} | {reward}\n")

                    state = next_state
                    if done or step_id >= 1600:
                        print(f"[Test] Episode {episode+1}, Test {test_i+1} ended. Reward: {total_test_reward}")
                        break
                f.write(f"Total Reward: {total_test_reward}\n")

            test_rewards_this_update.append(total_test_reward)

        avg_test_reward = np.mean(test_rewards_this_update)
        test_avg_rewards.append(avg_test_reward)

    env.close()

    # Save rewards to CSV
    csv_filename = "plot_values.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Training Reward", "Avg Test Reward"])
        for i in range(len(training_rewards)):
            writer.writerow([i+1, training_rewards[i], test_avg_rewards[i]])
    print(f"Saved plot values to {csv_filename}")

    # Plot training rewards with smoothing
    window_size = 5
    smoothed_training = smooth_data(training_rewards, window=window_size)
    training_x = np.arange(window_size - 1, len(training_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(training_rewards, label='Training Rewards', alpha=0.3, marker='o')
    plt.plot(training_x, smoothed_training, label='Smoothed Trend', color='blue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Episode Rewards (BipedalWalker)')
    plt.legend()
    plt.tight_layout()
    training_plot_filename = "training_rewards_smoothed.png"
    plt.savefig(training_plot_filename)
    print(f"Saved training rewards plot to {training_plot_filename}")
    plt.show()

    # Plot average test rewards with smoothing
    smoothed_test = smooth_data(test_avg_rewards, window=window_size)
    test_x = np.arange(window_size - 1, len(test_avg_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(test_avg_rewards, label='Avg Test Rewards', color='orange', alpha=0.3, marker='o')
    plt.plot(test_x, smoothed_test, label='Smoothed Trend', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (over 5 tests)')
    plt.title('Average Test Rewards (BipedalWalker)')
    plt.legend()
    plt.tight_layout()
    test_plot_filename = "avg_test_rewards_smoothed.png"
    plt.savefig(test_plot_filename)
    print(f"Saved average test rewards plot to {test_plot_filename}")
    plt.show()


if __name__ == "__main__":
    main()

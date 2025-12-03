import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt  
import copy

# ---------------------------
# LLM and Utility Imports
# ---------------------------
from openai import OpenAI
from google import genai
import json 
import csv  
from dotenv import load_dotenv

load_dotenv()

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
            "state (11 dimensions) | action (3 dimensions) | reward"
        """
        header = "state (11 dimensions) | action (3 dimensions) | reward\n"
        header += "---------------------------------------------------\n"
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
                "state": [float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x for x in s],
                "action": [float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x for x in a],
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
        state_bin_0 | ... | state_bin_10 | action_0 | action_1 | action_2
    """
    def __init__(self):
        # Each entry is a tuple: (state_tuple, action_tuple)
        self.table = []

    def to_string(self):
        """
        Return a string representation of the policy table with headers.
        """
        header = "z_pos | angle | thigh | leg | foot | x_vel | z_vel | angle_vel | thigh_vel | leg_vel | foot_vel | act_0 | act_1 | act_2\n"
        header += "-------------------------------------------------------------------------------------------------------------\n"
        lines = []
        for (s, a) in self.table:
            state_str = " | ".join(str(x) for x in s)
            action_str = " | ".join(str(x) for x in a)
            lines.append(f"{state_str} | {action_str}")
        return header + "\n".join(lines)

    def get_dict(self):
        """
        Convert the list of policy entries into a dictionary for easy lookup:
            dict[state_tuple] = action_tuple
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
                 num_buckets=None,
                 num_action_buckets=(3, 3, 3)):
        # Instantiate the LLM client
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        # Memory table for storing immediate rewards during an episode
        self.reward_table = MemoryTable()

        # Table mapping discretized states to best actions
        self.policy_table = PolicyTable()
        
        # Store the BEST policy seen so far to prevent catastrophic forgetting
        self.best_policy_table = PolicyTable()
        self.best_avg_test_reward = -float('inf')

        # List to store the conversation history with the LLM
        self.llm_conversation = []

        # Optimized Discretization for Hopper
        # We use fewer buckets for less critical dimensions to reduce state space size.
        # Total states: 2*3*2*2*2*2*1*2*1*1*1 = 384 states
        if num_buckets is None:
            self.num_buckets = (
                2,  # z-pos (Low, High)
                3,  # angle (Back, Upright, Forward)
                2,  # thigh (Flexed, Extended)
                2,  # leg
                2,  # foot
                2,  # x_vel (Slow, Fast)
                1,  # z_vel (Ignored)
                2,  # angle_vel (Stable, Spinning)
                1,  # thigh_vel (Ignored)
                1,  # leg_vel (Ignored)
                1   # foot_vel (Ignored)
            )
        else:
            self.num_buckets = num_buckets
        
        # Define bounds for each state dimension
        # Matches Hopper-v4/v5 typical ranges
        self.state_bounds = [
            (0.7, 1.5),       # z-coordinate (Fail < 0.7)
            (-0.2, 0.2),      # angle of torso (Fail > 0.2 rad)
            (-1.0, 1.0),      # angle of thigh joint
            (-1.0, 1.0),      # angle of leg joint
            (-1.0, 1.0),      # angle of foot joint
            (-1.0, 2.0),      # velocity of x-coordinate (Forward > 0)
            (-2.0, 2.0),      # velocity of z-coordinate
            (-2.0, 2.0),      # angular velocity of torso
            (-5.0, 5.0),      # angular velocity of thigh
            (-5.0, 5.0),      # angular velocity of leg
            (-5.0, 5.0)       # angular velocity of foot
        ]
        
        # Action discretization
        self.num_action_buckets = num_action_buckets
        self.action_bounds = [(-1.0, 1.0)] * 3  # All 3 actions are in range [-1, 1]

        # History of reward tables from episodes
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
        Convert the continuous state from the Hopper environment into a discrete state.
        """
        discrete_indices = []
        for i, val in enumerate(obs):
            low, high = self.state_bounds[i]
            # Clip the value to lie within the specified bounds
            clipped = max(min(val, high), low)
            ratio = (clipped - low) / (high - low)
            nb = self.num_buckets[i]
            # Map the continuous ratio to a discrete bucket index
            if nb == 1:
                idx = 0
            else:
                idx = int(round((nb - 1) * ratio))
                idx = max(0, min(idx, nb - 1))
            discrete_indices.append(idx)

        return tuple(discrete_indices)
    
    def discretize_action(self, continuous_action):
        """
        Convert a continuous action to a discretized action.
        """
        discrete_indices = []
        for i, val in enumerate(continuous_action):
            low, high = self.action_bounds[i]
            clipped = max(min(val, high), low)
            ratio = (clipped - low) / (high - low)
            nb = self.num_action_buckets[i]
            idx = int(round((nb - 1) * ratio))
            idx = max(0, min(idx, nb - 1))
            discrete_indices.append(idx)
            
        return tuple(discrete_indices)
    
    def continuous_action_from_discrete(self, discrete_action):
        """
        Convert a discretized action back to a continuous action.
        """
        continuous_action = []
        for i, idx in enumerate(discrete_action):
            low, high = self.action_bounds[i]
            nb = self.num_action_buckets[i]
            # Map the discrete index to a continuous value
            if nb <= 1:
                ratio = 0.5
            else:
                ratio = idx / (nb - 1)
            val = low + ratio * (high - low)
            continuous_action.append(val)
            
        return np.array(continuous_action)

    def get_action(self, state, env, use_best=False):
        """
        Determine the action to take given the current state.
        If use_best is True, uses the stored best policy.
        """
        disc_state = self.discretize_state(state)
        
        if use_best:
            policy_dict = self.best_policy_table.get_dict()
        else:
            policy_dict = self.policy_table.get_dict()
        
        # Return the learned action if this state exists in the policy, else random action
        if disc_state in policy_dict:
            return self.continuous_action_from_discrete(policy_dict[disc_state])
        else:
            return env.action_space.sample()

    def add_to_reward_table(self, state, action, reward):
        """
        Save the observed state, action, and immediate reward to the reward table.
        """
        disc_state = self.discretize_state(state)
        disc_action = self.discretize_action(action)
        self.reward_table.table.append((disc_state, disc_action, reward))

    def query_llm(self):
        """
        Send the current conversation (context + prompt) to the LLM and return its response.
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
                    print("Retrying...")
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    print(f"Failed with model: {model} after 5 attempts")
                    break
        
        raise Exception(f"{model} failed after 5 attempts.")

    def llm_update_policy(self):
        """
        Use the LLM to generate an updated policy based on the best reward histories.
        """
        self.reset_llm_conversation()

        if self.reward_history:
            sorted_reward_history = sorted(self.reward_history, key=lambda x: x[0], reverse=True)
            best_reward_history = sorted_reward_history[:5]
            reward_history_json = json.dumps([json.loads(entry[1]) for entry in best_reward_history], indent=2)
        else:
            reward_history_json = "[]"
        
        system_prompt = f"""
        You are an expert Reinforcement Learning agent controlling a Hopper robot in MuJoCo.
        Your goal is to make the hopper hop forward as fast as possible while maintaining balance (upright torso).
        
        STATE REPRESENTATION (Discretized):
        1. z_pos: 0=Low/Falling, 1=Good Height (>1.1m)
        2. angle: 0=Leaning Back, 1=Upright, 2=Leaning Forward
        3. thigh: 0=Flexed, 1=Extended
        4. leg: 0=Flexed, 1=Extended
        5. foot: 0=Flexed, 1=Extended
        6. x_vel: 0=Stopped/Back, 1=Moving Forward
        7. z_vel: (Ignored)
        8. angle_vel: 0=Stable/Back, 1=Spinning Forward
        9-11. Joint Vels: (Ignored)
        
        ACTIONS (3 dimensions, values 0-2):
        0=Retract (-1.0), 1=Neutral (0.0), 2=Extend (1.0)
        Dimensions: [Thigh, Leg, Foot]
        
        PHYSICS HINTS:
        - To hop forward: Extend the leg and foot (Action 2) when touching the ground to push off.
        - To maintain balance: Adjust the thigh angle. If leaning forward, pull thigh back (Action 0).
        - Recovery: If z_pos is Low (0), you need to extend leg/foot aggressively to stand up.
        - Cycle: A hopping gait involves rhythmic extension and retraction.
        """

        # Provide the CURRENT (potentially risky) policy to iterate on,
        # but we will only keep it if it beats the best.
        old_policy_str = self.policy_table.to_string()

        user_prompt = f"""
        ---------------------
        Reward Table History (best 5 episodes) in JSON format:
        {reward_history_json}

        ---------------------
        Old Policy Table:
        {old_policy_str}

        INSTRUCTIONS:
        1. Analyze the successful episodes to find patterns in (State -> Action) mapping.
        2. Generate a NEW Policy Table.
        3. Ensure you address the critical states:
           - If z_pos=0 (Low), actions should extend leg to recover.
           - If angle=2 (Forward), thigh should retract to balance.
        4. Output ONLY the policy rows - NO explanations, NO markdown code blocks, NO headers, NO extra text.

        FORMAT (output EXACTLY this format for each policy entry):
        z_pos | angle | thigh | leg | foot | x_vel | z_vel | angle_vel | thigh_vel | leg_vel | foot_vel | act_0 | act_1 | act_2
        
        IMPORTANT: Start your response immediately with the first policy line. Do not include any text before or after the policy lines.
        """

        self.add_llm_conversation(system_prompt, "system")
        self.add_llm_conversation(user_prompt, "user")

        new_policy_str = self.query_llm()

        try:
            policy_json = json.loads(new_policy_str)
            print("Successfully parsed JSON from LLM response.")
        except json.JSONDecodeError:
            print("LLM response is not valid JSON. Proceeding with text line parsing.")

        # Temporary table to hold the new candidate policy
        candidate_policy_table = []
        parsed_count = 0

        for line in new_policy_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if "z_pos" in line or "----" in line or "Note" in line:
                continue
                
            parts = line.split("|")
            if len(parts) == 14:
                try:
                    state_parts = [int(part.strip()) for part in parts[:11]]
                    action_parts = [int(part.strip()) for part in parts[11:]]
                    
                    state_tuple = tuple(state_parts)
                    action_tuple = tuple(action_parts)
                    
                    candidate_policy_table.append((state_tuple, action_tuple))
                    parsed_count += 1
                except ValueError:
                    print(f"Failed to parse line: {line}")
        
        print(f"Successfully parsed {parsed_count} policy entries from LLM response.")
        
        # Update the current policy table to this new candidate
        self.policy_table.table = candidate_policy_table


# --------------------------------------
# Helper Function: Data Smoothing for Plotting
# --------------------------------------
def smooth_data(data, window=5):
    return np.convolve(data, np.ones(window) / window, mode='valid')


# --------------------------------------
# Main Loop: Train and Test Policy using LLM Updates
# --------------------------------------
def main():
    env = gym.make('Hopper-v5', render_mode="human")
    
    # Using default optimized buckets defined in __init__
    llm_brain = LLMBrain()

    folder = './logs/hopper/'
    os.makedirs(folder, exist_ok=True)

    NUM_EPISODES = 100 
    
    training_rewards = []
    test_avg_rewards = []
    
    for episode in range(NUM_EPISODES):
        # 1. TRAINING PHASE
        # -----------------
        # We use the CURRENT policy (which might be a new experimental one from LLM)
        llm_brain.reward_table.table = []

        state, _ = env.reset()
        done = False
        truncated = False
        step_id = 0
        total_reward = 0

        episode_folder = os.path.join(folder, f"episode_{episode+1}")
        os.makedirs(episode_folder, exist_ok=True)
        train_file = os.path.join(episode_folder, "training_episode.txt")

        with open(train_file, 'w') as f:
            f.write(f"Episode {episode+1}\n")
            f.write("Step | State (continuous) | Discretized State | Action | Reward\n")

            while not done and not truncated:
                step_id += 1
                # Use the current policy for training exploration
                action = llm_brain.get_action(state, env, use_best=False)
                next_state, reward, done, truncated, info = env.step(action)

                total_reward += reward
                llm_brain.add_to_reward_table(state, action, reward)

                disc_st = llm_brain.discretize_state(state)
                disc_act = llm_brain.discretize_action(action)
                f.write(f"{step_id} | {state} | {disc_st} | {disc_act} | {reward}\n")

                state = next_state
                # Hopper typically lasts 1000 steps.
                if done or truncated or step_id >= 1000:
                    print(f"[Train] Ep {episode+1} ended. Total reward: {total_reward}")
                    break
            f.write(f"Total Reward: {total_reward}\n")
        
        training_rewards.append(total_reward)
        
        # Add to history for LLM context
        reward_snapshot = llm_brain.reward_table.to_json(total_reward)
        llm_brain.reward_history.append((total_reward, reward_snapshot))

        # 2. TESTING PHASE
        # ----------------
        # Evaluate the current policy to see if it's better than our Best Policy
        TEST_EPISODES = 5
        test_rewards_this_update = []
        
        for test_i in range(TEST_EPISODES):
            state, _ = env.reset()
            done = False
            truncated = False
            step_id = 0
            total_test_reward = 0
            
            test_file = os.path.join(episode_folder, f"testing_episode_{test_i+1}.txt")
            with open(test_file, 'w') as f:
                f.write(f"Testing Episode {test_i+1}\n")
                while not done and not truncated:
                    step_id += 1
                    action = llm_brain.get_action(state, env, use_best=False)
                    next_state, reward, done, truncated, info = env.step(action)
                    total_test_reward += reward
                    f.write(f"{step_id} | {reward}\n")
                    state = next_state
                    if done or truncated or step_id >= 1000:
                        break
                f.write(f"Total: {total_test_reward}\n")
            
            test_rewards_this_update.append(total_test_reward)

        avg_test_reward = np.mean(test_rewards_this_update)
        test_avg_rewards.append(avg_test_reward)
        
        print(f"Current Policy Avg Reward: {avg_test_reward:.2f} | Best So Far: {llm_brain.best_avg_test_reward:.2f}")

        # 3. POLICY UPDATE & RETENTION LOGIC
        # ----------------------------------
        if avg_test_reward > llm_brain.best_avg_test_reward:
            print(f"New Best Policy Found! Updating Best Policy (Reward: {avg_test_reward:.2f})")
            llm_brain.best_avg_test_reward = avg_test_reward
            llm_brain.best_policy_table = copy.deepcopy(llm_brain.policy_table)
        else:
            print(f"Current policy ({avg_test_reward:.2f}) is worse than best ({llm_brain.best_avg_test_reward:.2f}). Reverting to Best Policy for next update generation.")
            # Revert: The "current" policy becomes the Best Policy again, so the LLM
            # builds off the best stable version instead of the broken one.
            llm_brain.policy_table = copy.deepcopy(llm_brain.best_policy_table)

        # Generate the NEXT policy iteration using the LLM
        # (It will use the reverted 'best' policy as the 'Old Policy' in the prompt)
        llm_brain.llm_update_policy()

        # Save the (possibly reverted) policy table to file
        policy_file = os.path.join(episode_folder, "policy_table.txt")
        with open(policy_file, 'w') as f:
            f.write(llm_brain.policy_table.to_string())


    env.close()

    csv_filename = os.path.join(folder, "plot_values.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Training Reward", "Avg Test Reward"])
        for i in range(len(training_rewards)):
            writer.writerow([i+1, training_rewards[i], test_avg_rewards[i]])
    print(f"Saved plot values to {csv_filename}")

    window_size = 5
    if len(training_rewards) >= window_size:
        smoothed_training = smooth_data(training_rewards, window=window_size)
        training_x = np.arange(window_size - 1, len(training_rewards))

        plt.figure(figsize=(8, 4))
        plt.plot(training_rewards, label='Training Rewards', alpha=0.3, marker='o')
        plt.plot(training_x, smoothed_training, label='Smoothed Trend', color='blue', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Episode Rewards')
        plt.legend()
        plt.tight_layout()
        training_plot_filename = os.path.join(folder, "training_rewards_smoothed.png")
        plt.savefig(training_plot_filename)
        print(f"Saved training rewards plot to {training_plot_filename}")
        plt.show()

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
        test_plot_filename = os.path.join(folder, "avg_test_rewards_smoothed.png")
        plt.savefig(test_plot_filename)
        print(f"Saved average test rewards plot to {test_plot_filename}")
        plt.show()
    else:
        print("Not enough data to smooth.")

if __name__ == "__main__":
    main()

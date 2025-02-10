import gymnasium as gym
import numpy as np
import os
import time
import matplotlib.pyplot as plt  # for plotting
from google import genai

# -------------
# LLM IMPORTS
# -------------
from openai import OpenAI
import json  # for JSON parsing/formatting
import csv

API_KEY = os.getenv('apiKey')
BASE_URL = os.getenv('baseURL')
GEMINI_API_KEY = os.getenv('geminiApiKey')


# -----------------------------------------------------------
#  Reward Shaping Wrapper: Rewards based on decrease in distance to the goal.
# -----------------------------------------------------------
class RewardShapingDistanceWrapper(gym.Wrapper):
    def __init__(self, env, progress_scale=100, goal_position=0.5):
        """
        Wraps an environment to add a reward based on the change in distance to the goal.
        
        Args:
            env: The original gym environment.
            progress_scale: A scaling factor for the distance progress reward.
            goal_position: The x-coordinate position that defines the goal.
                           (For MountainCar-v0, the goal is typically at 0.5)
        """
        super(RewardShapingDistanceWrapper, self).__init__(env)
        self.progress_scale = progress_scale
        self.goal_position = goal_position
        self.last_distance = None

    def reset(self, **kwargs):
        self.last_distance = None
        state, info = self.env.reset(**kwargs)
        position, _ = state
        # Compute the initial distance from the goal.
        self.last_distance = self.goal_position - position
        return state, info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        position, _ = state
        # Current distance from the goal. (If the car overshoots, this may become negative.)
        current_distance = self.goal_position - position
        if self.last_distance is None:
            self.last_distance = current_distance

        # Compute the progress as the decrease in distance to the goal.
        # A positive value means the car is getting closer.
        progress = (self.last_distance - current_distance) * self.progress_scale
        self.last_distance = current_distance

        # Augment the default reward (-1 per step) with the progress-based bonus.
        shaped_reward = reward + progress
        return state, shaped_reward, done, truncated, info


# --------------------------------
#  MemoryTable for Rewards
# --------------------------------
class MemoryTable:
    """
    A simple table storing (state, action, reward). This will be passed
    to the LLM for updating the policy.
    """
    def __init__(self):
        self.table = []

    def to_string(self):
        """
        Convert to lines: "state (position, velocity) | action | reward"
        """
        header = "state (position, velocity) | action | reward\n"
        header += "-----------------------------------------\n"
        lines = []
        for (s, a, r) in self.table:
            lines.append(f"{s} | {a} | {r}")
        return header + "\n".join(lines)

    def to_json(self, total_reward=None):
        """
        Convert the reward table to a JSON-formatted string.
        Each entry is a dict with keys: state, action, reward.
        Optionally include the total_reward.
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


# --------------------------------
#  PolicyTable
# --------------------------------
class PolicyTable:
    """
    Stores a mapping from discrete states -> best action.
    The expected format from the LLM is:
       pos_bin | vel_bin | action
    """
    def __init__(self):
        self.table = []  # list of (state_tuple, action)

    def to_string(self):
        header = "pos_bin | vel_bin | action\n"
        header += "--------------------------\n"
        lines = []
        for (s, a) in self.table:
            (p, v) = s
            lines.append(f"{p} | {v} | {a}")
        return header + "\n".join(lines)

    def get_dict(self):
        d = {}
        for (s, a) in self.table:
            d[s] = a
        return d


# --------------------------------
#  LLMBrain
# --------------------------------
class LLMBrain:
    def __init__(self,
                 num_buckets=(20, 20),
                 env_action_space_n=3):
        # Uncomment and use your actual client if needed:
        # self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        self.reward_table = MemoryTable()
        self.policy_table = PolicyTable()
        self.llm_conversation = []

        # Discretization settings for MountainCar:
        # State dimensions: position and velocity.
        self.num_buckets = num_buckets
        self.state_bounds = [
            (-1.2, 0.6),    # position
            (-0.07, 0.07)   # velocity
        ]
        self.action_space_n = env_action_space_n

        # History of reward tables as tuples: (total_reward, reward_table_json_string)
        self.reward_history = []

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        self.llm_conversation.append({"role": role, "content": text})

    def discretize_state(self, obs):
        """
        Discretize the continuous MountainCar state into a 2D discrete state.
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
        If not found, return a random action.
        """
        disc_state = self.discretize_state(state)
        policy_dict = self.policy_table.get_dict()
        if disc_state in policy_dict:
            return policy_dict[disc_state]
        else:
            return env.action_space.sample()

    def add_to_reward_table(self, state, action, reward):
        disc_state = self.discretize_state(state)
        self.reward_table.table.append((disc_state, action, reward))

    def query_llm(self):
        """
        Send the conversation to the LLM and get the response text.
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
                    print("Retrying... Waiting for 60 seconds before retrying...")
                    time.sleep(60)
                else:
                    print(f"Failed with model: {model} after 5 attempts")
                    break
        raise Exception(f"{model} failed after 5 attempts.")

    def llm_update_policy(self):
        """
        Provide the reward history (best 5 episodes) and current policy to the LLM,
        and update the policy based on the LLM's response.
        """
        self.reset_llm_conversation()

        if self.reward_history:
            sorted_reward_history = sorted(self.reward_history, key=lambda x: x[0], reverse=True)
            best_reward_history = sorted_reward_history[:5]
            reward_history_json = json.dumps([json.loads(entry[1]) for entry in best_reward_history], indent=2)
        else:
            reward_history_json = "[]"
        
        system_prompt = f"""
You are an AI that decides a simple tabular policy for the MountainCar environment.
The environment has 2 discrete dimensions (position, velocity) and 3 possible actions (0: push left, 1: no push, 2: push right).
Reward tables are provided in JSON format.
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
pos_bin | vel_bin | action
        """

        self.add_llm_conversation(system_prompt, "system")
        self.add_llm_conversation(user_prompt, "user")

        new_policy_str = self.query_llm()

        print("LLM Response:")
        print(new_policy_str)

        try:
            policy_json = json.loads(new_policy_str)
            print("Successfully parsed JSON from LLM response.")
        except json.JSONDecodeError:
            print("LLM response is not valid JSON. Proceeding with text line parsing.")

        self.policy_table.table = []
        parsed_count = 0

        for line in new_policy_str.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 3:
                try:
                    pos_bin = int(parts[0].strip())
                    vel_bin = int(parts[1].strip())
                    act = int(parts[2].strip())
                    state = (pos_bin, vel_bin)
                    self.policy_table.table.append((state, act))
                    parsed_count += 1
                except ValueError:
                    print(f"Failed to parse line: {line}")
        print(f"Successfully parsed {parsed_count} policy entries from LLM response.")


# --------------------------------
#  Helper function for smoothing data
# --------------------------------
def smooth_data(data, window=5):
    return np.convolve(data, np.ones(window) / window, mode='valid')


# --------------------------------
#  MAIN LOOP
# --------------------------------
def main():
    # Create the MountainCar environment and wrap it with the reward shaping wrapper.
    base_env = gym.make('MountainCar-v0', render_mode=None)
    env = RewardShapingDistanceWrapper(base_env, progress_scale=100, goal_position=0.5)
    
    llm_brain = LLMBrain(num_buckets=(20, 20),
                         env_action_space_n=env.action_space.n)

    folder = './mountaincar_llm_reward_shaping_distance_logs/'
    os.makedirs(folder, exist_ok=True)

    NUM_EPISODES = 100

    # Lists for plotting rewards
    training_rewards = []
    test_avg_rewards = []  # Average reward over test episodes after each policy update

    for episode in range(NUM_EPISODES):
        # Clear the reward table for the new episode
        llm_brain.reward_table.table = []

        state, _ = env.reset()
        done = False
        step_id = 0
        total_reward = 0

        # Logging for training episode
        episode_folder = os.path.join(folder, f"episode_{episode+1}")
        os.makedirs(episode_folder, exist_ok=True)
        train_file = os.path.join(episode_folder, "training_episode.txt")

        with open(train_file, 'w') as f:
            f.write(f"Episode {episode+1}\n")
            f.write("Step | State (continuous) | Discretized State | Action | Reward\n")

            while not done:
                step_id += 1
                action = llm_brain.get_action(state, env)
                next_state, reward, done, truncated, info = env.step(action)

                total_reward += reward
                llm_brain.add_to_reward_table(state, action, reward)

                disc_st = llm_brain.discretize_state(state)
                f.write(f"{step_id} | {state} | {disc_st} | {action} | {reward}\n")

                state = next_state
                if done or step_id >= 200:
                    print(f"[Train] Episode {episode+1} ended. Total reward: {total_reward}")
                    break
            f.write(f"Total Reward: {total_reward}\n")
        
        training_rewards.append(total_reward)
        
        # Save the current episode's reward table and total reward as JSON.
        reward_snapshot = llm_brain.reward_table.to_json(total_reward)
        llm_brain.reward_history.append((total_reward, reward_snapshot))

        # Update the policy using the LLM.
        llm_brain.llm_update_policy()

        # Save the new policy to a file.
        policy_file = os.path.join(episode_folder, "policy_table.txt")
        with open(policy_file, 'w') as f:
            f.write(llm_brain.policy_table.to_string())

        # Test the updated policy.
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
                    next_state, reward, done, truncated, info = env.step(action)

                    total_test_reward += reward
                    disc_st = llm_brain.discretize_state(state)
                    f.write(f"{step_id} | {state} | {disc_st} | {action} | {reward}\n")

                    state = next_state
                    if done or step_id >= 200:
                        print(f"[Test] Episode {episode+1}, Test {test_i+1} ended. Total reward: {total_test_reward}")
                        break
                f.write(f"Total Reward: {total_test_reward}\n")

            test_rewards_this_update.append(total_test_reward)

        avg_test_reward = np.mean(test_rewards_this_update)
        test_avg_rewards.append(avg_test_reward)

    env.close()

    # -------------------
    #  Save Plot Values and Figures
    # -------------------
    csv_filename = "mountaincar_reward_shaping_distance_plot_values.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Episode", "Training Reward", "Avg Test Reward"])
        for i in range(len(training_rewards)):
            writer.writerow([i+1, training_rewards[i], test_avg_rewards[i]])
    print(f"Saved plot values to {csv_filename}")

    window_size = 5

    # Plot training rewards
    smoothed_training = smooth_data(training_rewards, window=window_size)
    training_x = np.arange(window_size - 1, len(training_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(training_rewards, label='Training Rewards', alpha=0.3, marker='o')
    plt.plot(training_x, smoothed_training, label='Smoothed Trend', color='blue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Episode Rewards (MountainCar with Distance-Based Reward Shaping)')
    plt.legend()
    plt.tight_layout()
    training_plot_filename = "mountaincar_training_rewards_distance_shaped.png"
    plt.savefig(training_plot_filename)
    print(f"Saved training rewards plot to {training_plot_filename}")
    plt.show()

    # Plot average test rewards
    smoothed_test = smooth_data(test_avg_rewards, window=window_size)
    test_x = np.arange(window_size - 1, len(test_avg_rewards))

    plt.figure(figsize=(8, 4))
    plt.plot(test_avg_rewards, label='Avg Test Rewards', color='orange', alpha=0.3, marker='o')
    plt.plot(test_x, smoothed_test, label='Smoothed Trend', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (over 5 tests)')
    plt.title('Average Test Rewards (MountainCar with Distance-Based Reward Shaping)')
    plt.legend()
    plt.tight_layout()
    test_plot_filename = "mountaincar_avg_test_rewards_distance_shaped.png"
    plt.savefig(test_plot_filename)
    print(f"Saved average test rewards plot to {test_plot_filename}")
    plt.show()


if __name__ == "__main__":
    main()

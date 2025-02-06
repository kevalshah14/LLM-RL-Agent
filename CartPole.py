import gymnasium as gym
from openai import OpenAI
import numpy as np
import os
import time
import json
from google import genai
import re
API_KEY = os.getenv('apiKey')
BASE_URL = os.getenv('baseURL')
GEMINI_API_KEY = os.getenv('geminiApiKey')

class MemoryTable:
    def __init__(self, role="reward"):
        assert role in ["reward", "q_value"]
        self.table = []
        self.role = role
    
    def to_string(self):
        table_template = f"""
state (x, dx, theta, dtheta) | action | {self.role}
----------------------------------------------
        """
        table = table_template
        for state, action, value in self.table:
            table += f"{state} | {action} | {value}\n"
        return table

    def get_dict(self):
        result = {}
        for state, action, value in self.table:
            if state not in result:
                result[state] = {}
            result[state][action] = value
        return result


class LLMBrain:
    def __init__(self):
        # Create the LLM client 
        # self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        # Memory tables
        self.reward_table = MemoryTable(role="reward")
        self.q_table = MemoryTable(role="q_value")
        
        # Conversation memory
        self.llm_conversation = []
        
        # Discretization bins for CartPole
        self.x_bins        = np.linspace(-2.4,   2.4,   10)  # cart position
        self.dx_bins       = np.linspace(-3.0,   3.0,   10)  # cart velocity
        self.theta_bins    = np.linspace(-0.21,  0.21,  10)  # pole angle (radians ~ -12 deg to +12 deg)
        self.dtheta_bins   = np.linspace(-3.0,   3.0,   10)  # pole angular velocity

    def discretize_state(self, state):
        """ 
        Discretize the 4D continuous state vector:
          state = [x, x_dot, theta, theta_dot]
        """
        x, dx, theta, dtheta = state
        
        x_idx      = np.digitize(x,      self.x_bins)      - (len(self.x_bins) // 2)
        dx_idx     = np.digitize(dx,     self.dx_bins)     - (len(self.dx_bins) // 2)
        theta_idx  = np.digitize(theta,  self.theta_bins)  - (len(self.theta_bins) // 2)
        dtheta_idx = np.digitize(dtheta, self.dtheta_bins) - (len(self.dtheta_bins) // 2)
        
        return (x_idx, dx_idx, theta_idx, dtheta_idx)
    
    def add_to_reward_table(self, state, action, reward):
        """ Add an experience to the reward table. """
        discretized_state = self.discretize_state(state)
        self.reward_table.table.append((discretized_state, action, reward))
    
    def get_reward_table_str(self):
        return self.reward_table.to_string()
    
    # LLM conversation management
    def reset_llm_conversation(self):
        self.llm_conversation = []
    
    def add_llm_conversation(self, text, role):
        self.llm_conversation.append({"role": role, "content": text})
    
    # def query_llm(self):
    #     models = [
    #         "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Top priority
    #         "mistralai/Mistral-7B-Instruct-v0.2",     # Secondary priority
    #         "meta-llama/Llama-3.3-70B-Instruct",      # Tertiary priority
    #     ]
        
    #     for model in models:
    #         for attempt in range(5):
    #             try:
    #                 print(f"Attempting with model: {model}")
    #                 completion = self.client.chat.completions.create(
    #                     model=model,
    #                     messages=self.llm_conversation,
    #                     max_tokens=2000  

    #                 )
    #                 response = completion.choices[0].message.content
    #                 self.add_llm_conversation(response, "assistant")
    #                 return response
    #             except Exception as e:
    #                 print(f"Error with model {model}: {e}")
    #                 if attempt < 4:
    #                     print("Retrying...")
    #                     print("Waiting for 120 seconds before retrying...")
    #                     time.sleep(120)
    #                 else:
    #                     print(f"Failed with model: {model}")
    #                     break  # Move to the next model if the current one fails after 5 attempts
        
    #     raise Exception("All models failed after 5 attempts each.")
    def query_llm(self):
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
    
    def get_action(self, state, env):
        """
        Return an action (0 or 1 in CartPole).
          0 = push left, 1 = push right
        """
        discretized_state = self.discretize_state(state)
        
        # If Q-table is missing data for this state, random action
        q_table_dict = self.q_table.get_dict()
        if discretized_state not in q_table_dict:
            return env.action_space.sample()
        
        # Otherwise, pick the action with the highest known Q-value
        best_action = None
        best_q_value = -np.inf
        
        for action in range(env.action_space.n):
            q_value = q_table_dict[discretized_state].get(action, -np.inf)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action
    
    def llm_update_q_table(self, episode_reward=None): 
        """
        Send the reward table and old Q-table to the LLM and ask for updated Q-values in JSON.
        """

        self.reset_llm_conversation()

        system_prompt = """
        You are an expert at Q-learning. Return updated Q-values for the CartPole problem strictly in JSON.
        Use the following structure verbatim:

        {
        "Q-Table": [
            {"state": [x_idx, dx_idx, theta_idx, dtheta_idx], "action": 0, "q_value": 0.0},
            ...
        ]
        }

        No extra keys, no explanations.
        """
        self.add_llm_conversation(system_prompt, role="system")

        user_prompt = f"""
        Here is the reward table:
        {self.reward_table.to_string()}

        Here is the old Q-table:
        {self.q_table.to_string()}

        Please produce the updated Q-table in the format shown in the system prompt.
        """
        if episode_reward is not None and episode_reward < 10:
                    user_prompt += "\n\nWARNING: The last episode had a low reward ({}). Focus on increasing "\
                                "Q-values for states where the pole is near falling (extreme angles/positions) "\
                                "and penalize dangerous states more heavily.".format(episode_reward)

        self.add_llm_conversation(user_prompt, role="user")


        llm_response = self.query_llm()
        print(f"LLM raw response:\n{llm_response}")

        # Define a regex pattern to extract the JSON content from a code block.
        # This pattern looks for a block starting with "```json", captures everything until the closing "```"
        pattern = r"```json\s*(\{.*?\})\s*```"

        # Use re.DOTALL so that '.' matches newline characters as well.
        match = re.search(pattern, llm_response, re.DOTALL)

        if match:
            json_str = match.group(1)  # The captured JSON string without the markdown formatting
            try:
                new_entries = json.loads(json_str)

                # 'new_entries' should be a dict with a key "Q-Table" that holds a list of entries.
                q_table_list = new_entries.get("Q-Table", [])
                for entry in q_table_list:
                    state_list = entry["state"]
                    action     = entry["action"]
                    q_value    = entry["q_value"]
                    # Convert the state list to a tuple
                    state_tuple = tuple(state_list)
                    # Add to Q-table memory
                    self.q_table.table.append((state_tuple, action, q_value))
            except json.JSONDecodeError:
                print("Error: Extracted JSON is invalid. No update performed.")
        else:
            print("Error: No JSON block found in the LLM response. No update performed.")



if __name__ == "__main__":
    # Create the CartPole environment
    env = gym.make('CartPole-v1', render_mode='human')

    llm_brain = LLMBrain()

    # Folder to store logs
    folder = './cartpole_logs/'
    os.makedirs(folder, exist_ok=True)

    # Number of training episodes
    num_episodes = 10

    for episode in range(num_episodes):
        state, _ = env.reset()
        llm_brain.reward_table.table = []  # reset reward table for this episode
        done = False
        total_reward = 0
        step_count = 0

        episode_folder = os.path.join(folder, f"episode_{episode+1}")
        os.makedirs(episode_folder, exist_ok=True)

        train_filename = os.path.join(episode_folder, "training_episode.txt")
        with open(train_filename, 'w') as f:
            f.write(f"Training Episode {episode+1}\n")
            f.write("Step | State | DiscretizedState | Action | Reward\n")

            while not done:
                step_count += 1
                action = llm_brain.get_action(state, env)
                next_state, reward, done, info, _ = env.step(action)

                total_reward += reward
                llm_brain.add_to_reward_table(state, action, reward)

                f.write(f"{step_count} | {state} | {llm_brain.discretize_state(state)} | {action} | {reward}\n")

                state = next_state
                if done or step_count >= 500:
                    print(f"Episode {episode + 1} finished with total reward: {total_reward}")
                    break

            f.write(f"Total reward: {total_reward}\n")
        
        # Query the LLM to update the Q-table based on the newly collected experience
        llm_brain.llm_update_q_table(episode_reward=total_reward)

        # Save the new Q-table to file
        q_table_filename = os.path.join(episode_folder, f"episode_{episode+1}_q_table.txt")
        with open(q_table_filename, 'w') as f:
            f.write(llm_brain.q_table.to_string())

        # ---- Testing / Evaluation ----
        # Run a few test episodes with the updated Q-table (no LLM updates here)
        for test_i in range(20):
            state, _ = env.reset()
            done = False
            total_reward = 0
            step_count = 0

            test_filename = os.path.join(episode_folder, f"testing_episode_{test_i+1}.txt")
            with open(test_filename, 'w') as f:
                f.write(f"Testing Episode {test_i+1}\n")
                f.write("Step | State | DiscretizedState | Action | Reward\n")

                while not done:
                    step_count += 1
                    action = llm_brain.get_action(state, env)
                    next_state, reward, done, info, _ = env.step(action)

                    total_reward += reward
                    llm_brain.add_to_reward_table(state, action, reward)

                    f.write(f"{step_count} | {state} | {llm_brain.discretize_state(state)} | {action} | {reward}\n")

                    state = next_state
                    if done or step_count >= 500:
                        print(f"Test {test_i+1}: total reward = {total_reward}")
                        break

                f.write(f"Total reward: {total_reward}\n")

    env.close()

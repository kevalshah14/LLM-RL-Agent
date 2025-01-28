import gymnasium as gym
from openai import OpenAI
import numpy as np
import os
import time

API_KEY = os.getenv('apiKey')
BASE_URL = os.getenv('baseURL')

class MemoryTable:
    def __init__(self, role="reward"):
        assert role in ["reward", "q_value"]
        self.table = []
        self.role = role
    
    def to_string(self):
        table_template = f"""
state | action | {self.role}
------------------------------
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
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.reward_table = MemoryTable(role="reward")
        self.q_table = MemoryTable(role="q_value")
        self.llm_conversation = []
        self.position_bins = np.linspace(-1.2, 0.6, 10)
        self.velocity_bins = np.linspace(-0.07, 0.07, 20)
    
    def discretize_state(self, state):
        position_idx = np.digitize(state[0], self.position_bins) - 5
        velocity_idx = np.digitize(state[1], self.velocity_bins) - 10
        return (position_idx, velocity_idx)
    
    def add_to_reward_table(self, state, action, reward):
        state = self.discretize_state(state)
        self.reward_table.table.append((state, action, reward))
    
    def get_reward_table_str(self):
        return self.reward_table.to_string()
    
    def reset_llm_conversation(self):
        self.llm_conversation = []
    
    def add_llm_conversation(self, text, role):
        self.llm_conversation.append({"role": role, "content": text})
    
    def query_llm(self):
        for attempt in range(5):
            try:
                completion = self.client.chat.completions.create(
                    #model="o1-preview",
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    messages=self.llm_conversation
                )
                # add the response to self.llm_conversation
                self.add_llm_conversation(completion.choices[0].message.content, "assistant")
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 4:
                    raise Exception("Failed")
                else:
                    print("Waiting for 120 seconds before retrying...")
                    time.sleep(120)
    
    def get_action(self, state, env):
        state = self.discretize_state(state)
        # Get the best action based on the Q-table
        # If the state is not in the Q-table, take a random action
        if state not in self.q_table.get_dict():
            return env.action_space.sample()
        else:
            # Get the best action based on the Q-table
            # If some actions are missing at a state, take a random action
            # If all actions are there, always take the actions which has the highest potential to have the highest q-value
            best_action = None
            best_q_value = -np.inf
            for action in range(env.action_space.n):
                q_value = self.q_table.get_dict()[state].get(action, -np.inf)
                if q_value > best_q_value:
                    best_action = action
                    best_q_value = q_value
            return best_action
    

    def llm_update_q_table(self):
        self.reset_llm_conversation()
        system_prompt = f"""
You are a smart expert whose goal is to synthesize a good q-table to guide the agent's behavior in the environment of mountain car.
The mountain car environment is a 2D world where the agent (car) can move left, stay still, or move right (corresponding to action 0, 1, 2). The goal is to reach the flag at the far right side of the hill. 
Next, you will see a table that shows the reward values for each state-action pair. Use this information to improve the q-table.
Note that the state is discretized into 10 bins for position (from -5 to 4) and 20 bins for velocity (from -10 to 9).
{self.reward_table.to_string()}

This reward table is generated based on the previous q-table:
{self.q_table.to_string()}

Based on the reward values, please provide a new q-table that you think will help the agent achieve its goal. Please generate the new q-table in the same format as the previous q-table (state, action, q_value, uncertainty).
        """
        self.add_llm_conversation(system_prompt, "user")
        new_q_table = self.query_llm()

        self.add_llm_conversation(new_q_table, "assistant")
        self.add_llm_conversation("Thank you for providing the new q-table. Please re-format it to lines of \"position | velocity | action | q_value\\n\"", "user")
        new_q_table = self.query_llm()

        print(f"New Q-table: {new_q_table}")
        
        # Update the Q-table based on the new Q-table
        for row in new_q_table.split("\n"):
            if row.strip():
                row = row.split("|")
                if len(row) == 4:
                    position, velocity, action, q_value = row
                    # state = state.strip()
                    # action = int(action.strip())
                    # q_value = float(q_value.strip())
                    # self.q_table.table.append((state, action, q_value))
        


                    try:
                        position = int(position.strip())
                        velocity = int(velocity.strip())
                        action = int(action.strip())
                        q_value = float(q_value.strip())
                        self.q_table.table.append(((position, velocity), action, q_value))
                    except:
                        pass
    



# Create the Mountain Car environment
env = gym.make('MountainCar-v0', render_mode=None, max_episode_steps=100)
llm_brain = LLMBrain()

# Initialize the environment
state = env.reset()

folder = './mountain_car_logs/non_explainable/'
os.makedirs(folder, exist_ok=True)

for episode in range(100):  # Run for 100 episodes
    state, _ = env.reset()
    llm_brain.reward_table.table = []
    done = False
    total_reward = 0
    step_id = 0
    reward = 0


    episode_folder = os.path.join('./mountain_car_logs/non_explainable/', str(episode + 1))
    os.makedirs(episode_folder, exist_ok=True)

    filename = os.path.join(episode_folder, f"training_episode.txt")
    with open(filename, 'w') as f:
        f.write(f"Episode {episode + 1}\n")
        f.write("Step | State | Discretized State | Action | Reward\n")

        while not done:
            step_id += 1

            # Select an action (0: push left, 1: no push, 2: push right)
            action = llm_brain.get_action(state, env)

            # Take the action and observe the result
            next_state, reward, done, info, _ = env.step(action)
            total_reward += reward
            llm_brain.add_to_reward_table(state, action, reward)
            f.write(f"{step_id} | {state} | {llm_brain.discretize_state(state)} | {action} | {reward}\n")

            # Update the state
            state = next_state

            if done or step_id >= 300:
                print(f"Episode {episode + 1} finished with total reward: {total_reward}")
                break
        
        f.write(f"Total reward: {total_reward}\n")
        f.close()

        

    # Use LLM to update the Q-table
    llm_brain.llm_update_q_table()

    # Write the q-table to file
    q_table_filename = os.path.join(episode_folder, f"episode_{episode + 1}_q_table.txt")
    with open(q_table_filename, 'w') as f:
        f.write(llm_brain.q_table.to_string())

    for testing_episode in range(20):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_id = 0
        reward = 0
        filename = os.path.join(episode_folder, f"testing_episode_{testing_episode + 1}.txt")
        with open(filename, 'w') as f:
            f.write(f"Episode {testing_episode + 1}\n")
            f.write("Step | State | Discretized State | Action | Reward\n")

            while not done:
                step_id += 1

                # Select an action (0: push left, 1: no push, 2: push right)
                action = llm_brain.get_action(state, env)

                # Take the action and observe the result
                next_state, reward, done, info, _ = env.step(action)
                total_reward += reward
                llm_brain.add_to_reward_table(state, action, reward)

                f.write(f"{step_id} | {state} | {llm_brain.discretize_state(state)} | {action} | {reward}\n")

                # Update the state
                state = next_state

                if done or step_id >= 300:
                    print(f"Episode {testing_episode + 1} finished with total reward: {total_reward}")
                    break
            
            f.write(f"Total reward: {total_reward}\n")
            f.close()


env.close()

## 1. JSON Formatting

**Purpose**:  
Using JSON formatting for exchanging data with the language model (LLM) ensures a consistent, machine-readable structure. This is especially useful when the LLM must return structured data (like Q-table entries).

**Relevant Snippet**:  
Inside `llm_update_q_table()`, the system prompt instructs the LLM to return a strict JSON format. 

```python
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
```

And the LLM is explicitly asked to provide updated Q-values in JSON, which is then parsed using Python’s `json.loads`:

```python
try:
    new_entries = json.loads(llm_response)
    q_table_list = new_entries["Q-Table"]
    ...
except json.JSONDecodeError:
    print("Error: LLM did not return valid JSON. No update performed.")
```

This ensures that any updates to the Q-table can be handled programmatically.

---

## 2. Using Featherless.ai to Access More Models

**Purpose**:  
`featherless.ai` enables you to easily route requests to multiple model providers. In the provided code, the model endpoints (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`) suggest usage of a platform that can access various foundation models.

**Relevant Snippet**:  
Within `query_llm()`, a list of models is defined:

```python
models = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Top priority
    "mistralai/Mistral-7B-Instruct-v0.2",     # Secondary priority
    "meta-llama/Llama-3.3-70B-Instruct",      # Tertiary priority
]
```

---

## 3. Model Switching

**Purpose**:  
If one model is down or returns errors, you can automatically switch to another model to maintain system reliability.

**Relevant Snippet**:  
Inside `query_llm()`, we loop through each model and handle possible exceptions:

```python
for model in models:
    for attempt in range(5):
        try:
            print(f"Attempting with model: {model}")
            completion = self.client.chat.completions.create(
                model=model,
                messages=self.llm_conversation,
                max_tokens=2000
            )
            response = completion.choices[0].message.content
            self.add_llm_conversation(response, "assistant")
            return response
        except Exception as e:
            print(f"Error with model {model}: {e}")
            if attempt < 4:
                print("Retrying...")
                print("Waiting for 120 seconds before retrying...")
                time.sleep(120)
            else:
                print(f"Failed with model: {model}")
                break
```

If a model fails, the code attempts multiple retries. After exhausting retries, it moves on to the next model in the list. If all models fail, an exception is raised.

---

## 4. Performance-Based Prompting

**Purpose**:  
You can adapt your prompts based on the performance of the last episode. If the reward is below a certain threshold, extra guidance is appended to the prompt to help the LLM focus on specific improvements.

**Relevant Snippet**:  
In `llm_update_q_table()`, we check the `episode_reward`:

```python
if episode_reward is not None and episode_reward < 10:
    user_prompt += "\n\nWARNING: The last episode had a low reward ...
```

This appended text urges the LLM to prioritize fixing the Q-values associated with lower rewards or unstable states.

---

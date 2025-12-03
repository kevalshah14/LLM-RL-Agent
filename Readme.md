# LLM-RL-Agent

This project implements reinforcement learning agents for various environments using large language models (LLMs).

## Project Structure

```
agents/
    cartpole/
        cartpole_agent.py
    frozen_lake/
        frozen_lake_agent.py
    bipedal_walker/
        bipedal_walker_agent.py
    hopper/
        hopper_agent.py
logs/
    cartpole/
    frozen_lake/
    bipedal_walker/
    hopper/
pyproject.toml
README.md
```

## Requirements

- Python 3.11
- Gymnasium
- OpenAI
- NumPy
- Matplotlib
- Google GenAI
- Stable Baselines3

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kevalshah14/LLM-RL-Agent.git
    cd llm-rl-agent
    ```

2. Install dependencies (using `uv` or `pip`):
    ```sh
    uv sync
    # or
    pip install .
    ```

## Usage

### CartPole

To run the CartPole agent:
```sh
python agents/cartpole/cartpole_agent.py
```

### FrozenLake

To run the FrozenLake agent:
```sh
python agents/frozen_lake/frozen_lake_agent.py
```

### BipedalWalker

To run the BipedalWalker agent:
```sh
python agents/bipedal_walker/bipedal_walker_agent.py
```

### Hopper

To run the Hopper agent:
```sh
python agents/hopper/hopper_agent.py
```

## Project Details

### Logging

All agents log their results in the `logs/` directory. Each agent has its own subdirectory (e.g., `logs/cartpole/`). Within each agent's log directory, episodes are stored in numbered subfolders (e.g., `episode_1/`), containing training logs, policy tables, and testing results.

Plots and CSV summaries are also saved in the respective agent's log directory.

## License

This project is licensed under the MIT License.

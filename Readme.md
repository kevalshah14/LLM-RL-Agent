# LLM-RL-Agent

This project implements reinforcement learning agents for various environments using large language models (LLMs).

## Project Structure

```
.env
.gitignore
BipeadalWalker/
    BipedalWalker.py
Cartpole/
    cartpole_v2.py
    cartpole_v3.py
    CartPole_Without_LLM.py
    CartPole.py
    Improvements.md
FrozenLake/
    frozenlake_avg_test_rewards_smoothed1.png
    frozenlake_plot_values.csv
    frozenlake_training_rewards_smoothed1.png
    FrozenLake.py
MountainCar/
    MountainCar_v2.py
    MountainCar.py
poetry.lock
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

2. Install dependencies using Poetry:
    ```sh
    poetry install
    ```

## Usage

### CartPole

To run the CartPole agent:
```sh
python Cartpole/CartPole.py
```

### MountainCar

To run the MountainCar agent:
```sh
python MountainCar/MountainCar.py
```

### FrozenLake

To run the FrozenLake agent:
```sh
python FrozenLake/FrozenLake.py
```

### BipedalWalker

To run the BipedalWalker agent:
```sh
python BipeadalWalker/BipedalWalker.py
```

## Project Details

### CartPole

The CartPole agent logs its Q-table and testing results in the `cartpole_logs` directory. Each episode has its own subdirectory containing the Q-table and testing results.

### MountainCar

The MountainCar agent logs its results in the `mountain_car_logs` directory.

### FrozenLake

The FrozenLake agent logs its results in the `frozenlake_logs` directory.

### BipedalWalker

The BipedalWalker agent logs its results in the `bipedalwalker_logs` directory.

## License

This project is licensed under the MIT License.
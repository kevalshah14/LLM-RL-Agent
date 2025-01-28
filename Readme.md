# LLM-RL-Agent

This project implements reinforcement learning agents for various environments using large language models (LLMs).


## Requirements

- Python 3.11
- Gymnasium
- OpenAI
- NumPy

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
python CartPole.py
```

### MountainCar

To run the MountainCar agent:
```sh
python MountainCar.py
```

## Project Details

### CartPole

The CartPole agent logs its Q-table and testing results in the `cartpole_logs` directory. Each episode has its own subdirectory containing the Q-table and testing results.

### MountainCar

The MountainCar agent logs its results in the `mountain_car_logs` directory.

## License
This project is licensed under the MIT License.
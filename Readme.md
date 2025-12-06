# 🤖 LLM-RL-Agent

> **Leveraging Large Language Models as Policy Generators for Reinforcement Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![Gemini](https://img.shields.io/badge/Gemini-API-orange.svg)](https://ai.google.dev/)

A novel framework that uses **Large Language Models (LLMs)** as policy generators for reinforcement learning tasks. Instead of traditional gradient-based optimization, this approach leverages the reasoning capabilities of LLMs to directly generate and iteratively refine tabular policies.

## 📋 Overview

LLM-RL-Agent explores a fundamentally different approach to reinforcement learning:

- 🧠 **LLM as Policy Generator**: Uses Google's Gemini models to analyze experience and propose improved state-action mappings
- 📊 **Tabular Policies**: Generates interpretable, human-readable policy tables
- 🔄 **Iterative Refinement**: Learns from best/worst episodes to continuously improve
- 🛡️ **Catastrophic Forgetting Prevention**: Maintains best policy to ensure stable learning

## 🎮 Supported Environments

| Environment | State Space | Action Space | Discretization | Performance |
|------------|-------------|--------------|----------------|-------------|
| **CartPole-v1** | 4D continuous | Discrete (2) | 324 states | ✅ High |
| **FrozenLake-v1** | Discrete (16) | Discrete (4) | 16 states | ✅ High |
| **Hopper-v5** | 11D continuous | 3D continuous | 384 states | ⚠️ Limited |
| **BipedalWalker-v3** | 24D continuous | 4D continuous | 10,000 states | ⚠️ Limited |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LLM-RL-Agent                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Memory    │    │   Policy    │    │  LLM Brain  │  │
│  │   Table     │───▶│   Table     │◀───│  (Gemini)   │  │
│  │ (s, a, r)   │    │  s → a      │    │             │  │
│  └─────────────┘    └─────────────┘    └─────────────┘  │
│         │                 │                   │         │
│         ▼                 ▼                   ▼         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Gymnasium Environment              │    │
│  │   (CartPole / FrozenLake / Hopper / Bipedal)    │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kevalshah14/LLM-RL-Agent.git
   cd LLM-RL-Agent
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -e .
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   geminiApiKey=your_gemini_api_key_here
   ```
   
   Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### Running Agents

```bash
# CartPole (recommended to start)
python agents/cartpole/cartpole_agent.py

# FrozenLake
python agents/frozen_lake/frozen_lake_agent.py

# Hopper (requires MuJoCo)
python agents/hopper/hopper_agent.py

# BipedalWalker
python agents/bipedal_walker/bipedal_walker_agent.py
```

## 📁 Project Structure

```
LLM-RL-Agent/
├── agents/
│   ├── cartpole/
│   │   ├── cartpole_agent.py      # CartPole implementation
│   │   ├── Training_Logs.png      # Training visualization
│   │   └── Testing_Logs.png       # Testing visualization
│   ├── frozen_lake/
│   │   ├── frozen_lake_agent.py   # FrozenLake implementation
│   │   └── *.png                  # Result visualizations
│   ├── hopper/
│   │   └── hopper_agent.py        # Hopper implementation
│   └── bipedal_walker/
│       └── bipedal_walker_agent.py # BipedalWalker implementation
├── logs/                          # Training logs (generated)
├── paper/
│   └── llm_rl_agent_paper.tex     # Research paper (LaTeX)
├── pyproject.toml                 # Project configuration
├── .env                           # API keys (create this)
└── README.md
```

## 🔧 How It Works

### 1. State Discretization
Continuous states are discretized into bins for tabular representation:
```python
# Example: CartPole uses (3, 3, 6, 6) = 324 discrete states
# (position_bins, velocity_bins, angle_bins, angular_velocity_bins)
```

### 2. Episode Collection
The agent interacts with the environment, storing `(state, action, reward)` tuples.

### 3. LLM Policy Update
The best episodes are formatted as JSON and sent to the LLM with physics hints:
```
PHYSICS HINT:
- If the pole is leaning LEFT, push LEFT (Action 0)
- If the pole is leaning RIGHT, push RIGHT (Action 1)
```

### 4. Policy Parsing
The LLM's response is parsed into a new policy table:
```
pos_bin | vel_bin | angle_bin | angvel_bin | action
0       | 1       | 2         | 1          | 0
...
```

### 5. Best Policy Retention
If the new policy underperforms, the agent reverts to the best known policy.

## 📊 Results

### CartPole-v1
- **Max Reward**: 500 (maximum possible)
- **Convergence**: ~20-30 episodes
- **Success**: LLM effectively learns balancing physics

### FrozenLake-v1
- **Goal Reach Rate**: High (deterministic setting)
- **Convergence**: ~10-15 episodes
- **Success**: LLM reasons about spatial navigation well

### Hopper-v5 & BipedalWalker-v3
- **Performance**: Limited
- **Challenges**: 
  - Continuous control requires fine-grained actions
  - Temporal coordination for walking gaits
  - Large state spaces lead to sparse policies

## 🧪 Key Findings

| Aspect | Discrete Control | Continuous Control |
|--------|------------------|-------------------|
| Sample Efficiency | ✅ Excellent | ⚠️ Moderate |
| Policy Quality | ✅ Near-optimal | ❌ Limited |
| Interpretability | ✅ High | ✅ High |
| Temporal Reasoning | ⚠️ Limited | ❌ Poor |

**Best Use Cases:**
- Low-dimensional discrete/hybrid control
- Tasks with clear physics intuitions
- Educational/prototyping purposes

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{shah2024llmrlagent,
  title={LLM-RL-Agent: Leveraging Large Language Models as Policy Generators for Reinforcement Learning},
  author={Shah, Keval Rajesh},
  year={2024},
  institution={Arizona State University},
  url={https://github.com/kevalshah14/LLM-RL-Agent}
}
```

## 🤝 Contributing

Contributions are welcome! Some ideas:
- Support for additional environments
- Alternative LLM backends (OpenAI, Claude, etc.)
- Hierarchical policy generation
- Hybrid approaches with neural networks

## 📄 License

This project is for educational and research purposes.

## 👤 Author

**Keval Rajesh Shah**  
School of Computing and Augmented Intelligence  
Arizona State University  
📧 kshah57@asu.edu

---

<p align="center">
  <i>Exploring the intersection of Large Language Models and Reinforcement Learning</i>
</p>

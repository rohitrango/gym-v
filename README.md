<h1 align="center">Gym-V</h1>

<p align="center">
  <b>A Unified Vision Environment System for Agentic Vision Research</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#environment-catalogue">Environments</a> &bull;
  <a href="#key-findings">Key Findings</a>
</p>

---

**Gym-V** is a unified platform of **179 procedurally generated visual environments** across 10 domains with controllable difficulty, built on a Gymnasium-compatible API. It unifies interactive training, offline supervision, and benchmark evaluation under one interface — enabling controlled experiments on vision-language agents that were previously infeasible across fragmented toolkits.

### Highlights

- **179 environments** spanning single-turn reasoning, multi-turn games, spatial navigation, and retro arcade games
- **Gymnasium-compatible API** with multi-agent support, composable wrappers, and tool integration
- **Controllable difficulty** via parametric generation with difficulty presets (levels 0, 1, 2)
- **Evaluation-as-a-Service** with a distributed reward server (Ray Serve) supporting heterogeneous backends
- **Composable observation wrappers** that make task representation an explicit experimental variable

## Installation

```bash
# Basic installation
pip install -e .

# With optional environment groups
pip install -e ".[games]"       # Board/card games (TextArena, PettingZoo)
pip install -e ".[spatial]"     # 2D/3D navigation (MiniGrid, MiniWorld)
pip install -e ".[temporal]"    # Retro games (stable-retro)
pip install -e ".[vlmeval]"     # VLM evaluation benchmarks

# All optional dependencies
pip install -e ".[games,spatial,temporal,vlmeval,reasoning-gym]"
```

## Quick Start

```python
import gym_v

# Single-turn: observe an image, give an answer, receive a reward
env = gym_v.make("Arc/ArcAgi-v0")
obs, info = env.reset(seed=42)
# obs = {"agent_0": Observation(image=PIL.Image, text="...", metadata={})}
obs, reward, terminated, truncated, info = env.step({"agent_0": "[[0,1],[1,0]]"})
env.close()

# Multi-turn: interact with the environment over multiple steps
env = gym_v.make("Games/Chess-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step({"agent_0": "e2e4"})
# Continue stepping until terminated["__all__"] or truncated["__all__"]
env.close()
```

### Interactive Demo

```bash
python examples/demo.py --id "Games/TicTacToe-v0"
```

## Architecture

```
gym_v/
├── core.py              # Env, Observation, Wrapper base classes
├── envs/
│   ├── registration.py  # register() / make() system
│   ├── single_turn/     # 125 single-step reasoning environments
│   ├── multi_turn/      # 74 interactive environments
│   │   ├── games/       #   Board, card & puzzle games
│   │   ├── spatial/     #   2D/3D navigation tasks
│   │   └── temporal/    #   Retro arcade games (stable-retro)
│   ├── offline/         # Generic JSONL dataset loader
│   └── eval/            # VLMEval & GenEval integration
├── wrappers/            # Composable observation/action wrappers
├── tools/               # Agent tool system (IPython, etc.)
└── utils/               # Image, seeding, rendering utilities
```


## Key Findings

Using Gym-V, our experiments reveal several insights for training vision-language agents:

1. **Observation scaffolding > RL algorithm choice.** Captions, game rules, and interaction history determine whether learning succeeds at all — more so than the choice between GRPO, GSPO, or SAPO.

2. **Diverse training generalizes; narrow training hurts.** Cross-domain curricula transfer broadly, while training on a single domain can cause negative transfer. Multi-turn interaction amplifies both effects.

3. **RL closes the gap.** A 7B model trained with RL on Gym-V environments can surpass much larger models' zero-shot performance on several task categories.

For full results, see our paper.

## License

This project is for research use. See [LICENSE](LICENSE) for details.

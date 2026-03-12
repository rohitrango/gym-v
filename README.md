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

### Core Interface

Every environment follows the standard Gymnasium protocol:

| Method | Description |
|--------|-------------|
| `reset(seed=None)` | Returns `(obs_dict, info_dict)` |
| `step(action_dict)` | Returns `(obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict)` |
| `render()` | Returns a PIL Image of the current state |
| `close()` | Cleans up resources |

**Observations** are `Observation(image, text, metadata)` namedtuples. **Actions** are `{agent_id: action_string}` dicts.

## Environment Catalogue

**Total: 179 environments** — 105 Single-Turn, 74 Multi-Turn

---

### Single-Turn Environments (105)

Single-turn environments require a single interaction: the agent observes an image and provides an answer.

#### Arc (3)

Abstract Reasoning Corpus — infer transformation rules from input-output grid pairs.

| Environment ID | Description |
|----------------|-------------|
| `Arc/Arc1D-v0` | 1D ARC reasoning: infer transformation rules from sequences |
| `Arc/ArcAgi-v0` | ARC-AGI benchmark: 2D grid color transformation reasoning |
| `Arc/ReArc-v0` | Procedurally generated ARC tasks with unlimited instances |

#### Algorithmic (21)

Algorithm and simulation tasks covering matrix operations, game theory, search, and counting.

| Environment ID | Description |
|----------------|-------------|
| `Algorithmic/AdditionTable-v0` | Fill in missing values in a partial addition table |
| `Algorithmic/BinaryMatrix-v0` | Predict the next state of a binary matrix transformation |
| `Algorithmic/BinaryTreeLeafNumExpectation-v0` | Compute expected leaf count of a random binary tree |
| `Algorithmic/CirculatingGrid-v0` | Minimize edits to satisfy row/column constraints on a cyclic grid |
| `Algorithmic/CoinSquareGame-v0` | Coin-square combinatorial game: determine the winner |
| `Algorithmic/FaceRightWay-v0` | Compute minimum flips to align all elements in one direction |
| `Algorithmic/GameOfLife-v0` | Predict grid state after N steps of Conway's Game of Life |
| `Algorithmic/GraMinimaGame-v0` | Graph minima game: strategic analysis on a grid |
| `Algorithmic/GridBFS-v0` | Find the shortest path on a grid via BFS |
| `Algorithmic/GridLocalMinimumCounting-v0` | Count local minima in a grid |
| `Algorithmic/LandformGenerationCounting-v0` | Count valid terrain configurations under height constraints |
| `Algorithmic/LangtonAnt-QA-v0` | Predict Langton's Ant behavior after N steps |
| `Algorithmic/Lifegame-QA-v0` | Analyze Game of Life grid evolution |
| `Algorithmic/MatrixPermutationBothDiagonalOne-v0` | Count permutations placing 1s on both diagonals |
| `Algorithmic/MatrixPermutationMainDiagonalOne-v0` | Count permutations placing 1s on the main diagonal |
| `Algorithmic/MaxGridPathIntersection-v0` | Maximize path intersection points on a grid |
| `Algorithmic/MonochromeBlockCounting-v0` | Count monochromatic blocks in a grid |
| `Algorithmic/RotateMatrix-v0` | Output the rotated matrix |
| `Algorithmic/RottenOranges-v0` | Simulate orange decay propagation and compute total time |
| `Algorithmic/SpiralMatrix-v0` | Output matrix elements in spiral order |
| `Algorithmic/StoneGame-v0` | Stone-taking game: determine the winner |
| `Algorithmic/StoneIntervalsGame-v0` | Stone interval game: strategic analysis |
| `Algorithmic/TuringMachine2d-QA-v0` | Predict 2D Turing machine execution results |

#### Cognition (10)

Cognitive and spatial reasoning tasks including pattern recognition and 3D understanding.

| Environment ID | Description |
|----------------|-------------|
| `Cognition/Hue-QA-v0` | Color matching game Q&A |
| `Cognition/Maze3D-QA-v0` | 3D maze spatial reasoning Q&A |
| `Cognition/OddOneOutPoly-v0` | Identify the polygon that doesn't belong |
| `Cognition/RectangleCount-v0` | Count rectangles in a figure |
| `Cognition/RubiksCube-QA-v0` | Analyze Rubik's Cube state after rotations |
| `Cognition/SequenceCompletionPoly-v0` | Complete a polygon sequence by inferring the pattern |
| `Cognition/SymmetryFillPoly-v0` | Complete a symmetric figure |
| `Cognition/TransformResultPoly-v0` | Predict the result of geometric transformations |

#### Geometry (8)

Computational geometry tasks involving convex hulls, areas, and enclosing shapes.

| Environment ID | Description |
|----------------|-------------|
| `Geometry/ConvexHull-v0` | Compute the convex hull vertices of a point set |
| `Geometry/LargestRectangleAmongPoints-v0` | Find the largest rectangle area among points |
| `Geometry/PipelineArrangement-v0` | Optimize pipeline layout |
| `Geometry/SkaRockGarden-v0` | Japanese rock garden layout problem |
| `Geometry/SmallestCircle-v0` | Find the minimum enclosing circle for a point set |
| `Geometry/SumTriangleArea-v0` | Compute total triangle area |
| `Geometry/Tangram-QA-v0` | Tangram puzzle Q&A |
| `Geometry/VisibleLine-v0` | Determine visible line segments from above |

#### Graphs (23)

Graph theory tasks covering shortest paths, spanning trees, coloring, matching, and isomorphism.

| Environment ID | Description |
|----------------|-------------|
| `Graphs/FbiBinaryTree-v0` | FBI binary tree structure analysis |
| `Graphs/GraphContainTreeCounting-v0` | Count spanning trees in a graph |
| `Graphs/GraphIsomorphism-v0` | Determine graph isomorphism |
| `Graphs/GridComponent-v0` | Count connected components in a grid |
| `Graphs/HamiltonianPath-v0` | Find a Hamiltonian path |
| `Graphs/HamiltonianPathExistence-v0` | Determine Hamiltonian path existence |
| `Graphs/LargestIsland-v0` | Find the largest island area |
| `Graphs/LongestPath-v0` | Find the longest path in a graph |
| `Graphs/MaximumAchromaticNumber-v0` | Compute maximum achromatic number |
| `Graphs/MaximumClique-v0` | Find the maximum clique |
| `Graphs/MaximumIndependentSetGrid-v0` | Maximum independent set on a grid graph |
| `Graphs/MaximumIndependentSetTree-v0` | Maximum weighted independent set on a tree |
| `Graphs/MaximumWeightMatching-v0` | Find maximum weight matching |
| `Graphs/MinimumChromaticNumber-v0` | Compute minimum chromatic number |
| `Graphs/MinimumDirectedSpanningTree-v0` | Find minimum directed spanning tree |
| `Graphs/MinimumSpanningTreeCounting-v0` | Count minimum spanning trees |
| `Graphs/MixedGraphEulerianCircuit-v0` | Determine Eulerian circuit in a mixed graph |
| `Graphs/Patrol-v0` | Patrol route planning |
| `Graphs/ShortestPath-v0` | Weighted graph shortest path |
| `Graphs/SpyNetwork-v0` | Minimum vertex cover in a spy network |
| `Graphs/TreeCenter-v0` | Find the center of a tree |
| `Graphs/TreeChangeOneEdgeDiameter-v0` | Minimize tree diameter by changing one edge |
| `Graphs/TreeColoring-v0` | Count tree coloring configurations |
| `Graphs/TreeDistanceEqualTriadCounting-v0` | Count equidistant triads in a tree |
| `Graphs/TreeEvenPartitioning-v0` | Even partitioning of a tree |
| `Graphs/TreeTopologicalSequenceCounting-v0` | Count tree topological orderings |
| `Graphs/WeightedBinarytree-v0` | Maximize score on a weighted binary tree |

#### Logic (17)

Logic reasoning and constraint satisfaction puzzles.

| Environment ID | Description |
|----------------|-------------|
| `Logic/Binairo-v0` | Binary puzzle: fill 0/1 to satisfy row/column constraints |
| `Logic/BinarioNoAdjacencyRequirement-v0` | Binary fill puzzle without adjacency constraints |
| `Logic/CampsitePuzzle-v0` | Place tents next to trees following constraints |
| `Logic/CircuitLogic-v0` | Solve logic gate circuits |
| `Logic/Futoshiki-v0` | Inequality Sudoku variant |
| `Logic/GridParityConstruction-v0` | Grid parity construction puzzle |
| `Logic/HitoriPuzzle-v0` | Hitori puzzle variant |
| `Logic/Kakurasu-v0` | Kakurasu addition logic puzzle |
| `Logic/MagicSquarePuzzle-v0` | Fill in a magic square |
| `Logic/MiniSudoku-v0` | Mini Sudoku (4×4) |
| `Logic/NQueens-v0` | N-Queens placement problem |
| `Logic/Numbrix-v0` | Number path connection puzzle |
| `Logic/Renzoku-v0` | Consecutive number puzzle |
| `Logic/SkyscraperPuzzle-v0` | Skyscraper puzzle |
| `Logic/SkyscraperSumPuzzle-v0` | Skyscraper sum puzzle |
| `Logic/StarBattle-QA-v0` | Star Battle puzzle Q&A |
| `Logic/Survo-v0` | Survo logic puzzle |
| `Logic/Tents-QA-v0` | Tents puzzle Q&A |
| `Logic/Thermometers-v0` | Thermometer puzzle |

#### Perception

Chart and graph visual understanding tasks.

| Environment ID | Description |
|----------------|-------------|
| `Perception/3DReconstruction-QA-v0` | 3D reconstruction visual Q&A |
| `Perception/ChartToTable-v0` | Extract tabular data from charts |
| `Perception/ContourPlot-v0` | Understand functions from contour plots |
| `Perception/DAGToTopoOrder-v0` | Topological sort of a DAG |
| `Perception/FlowNetwork-v0` | Maximum flow analysis on a flow network |
| `Perception/FunctionGraph-v0` | Identify functions from their graphs |
| `Perception/GraphToAdjacency-v0` | Convert graph visualization to adjacency matrix |
| `Perception/GraphToMST-v0` | Find MST from a graph visualization |
| `Perception/ParametricCurve-v0` | Identify parametric curve equations |
| `Perception/PolarPlot-v0` | Identify polar coordinate equations |
| `Perception/TreeToTraversal-v0` | Produce tree traversal sequences |
| `Perception/VectorField-v0` | Identify vector field functions |

#### Puzzles (23)

Classic games and visual puzzle Q&A.

| Environment ID | Description |
|----------------|-------------|
| `Puzzles/ChessRanger-QA-v0` | Chess piece attack range Q&A |
| `Puzzles/EightDigitPuzzle-v0` | 8-puzzle: find the solution sequence |
| `Puzzles/Freecell-QA-v0` | FreeCell solitaire Q&A |
| `Puzzles/Jewel2-QA-v0` | Jewel match Q&A |
| `Puzzles/KloBlocks-v0` | Block elimination puzzle |
| `Puzzles/Klondike-QA-v0` | Klondike solitaire Q&A |
| `Puzzles/KnightSwap-v0` | Swap knight positions |
| `Puzzles/Maze-QA-v0` | Maze Q&A |
| `Puzzles/NinePuzzle-v0` | 9-puzzle sliding tile puzzle |
| `Puzzles/Pacman-QA-v0` | Pac-Man Q&A |
| `Puzzles/PyramidChess-QA-v0` | Pyramid chess Q&A |
| `Puzzles/RhythmGame-QA-v0` | Rhythm game Q&A |
| `Puzzles/Snake-QA-v0` | Snake game Q&A |
| `Puzzles/SpaceInvaders-QA-v0` | Space Invaders Q&A |
| `Puzzles/SpiderSolitaire-QA-v0` | Spider solitaire Q&A |
| `Puzzles/Tetris-QA-v0` | Tetris Q&A |
| `Puzzles/TetrisAttack-v0` | Tetris Attack: minimize swaps to clear blocks |
| `Puzzles/TicTacToe-QA-v0` | Tic-Tac-Toe Q&A |
| `Puzzles/TowerOfHanoi-v0` | Tower of Hanoi: find optimal move sequence |
| `Puzzles/Tsumego-v0` | Go life-and-death puzzle (Tsumego) |
| `Puzzles/TwiddlePuzzle-v0` | Twiddle puzzle: rotate sub-matrices to restore order |
| `Puzzles/UltraTicTacToe-QA-v0` | Ultimate Tic-Tac-Toe Q&A |
| `Puzzles/WordSearch-QA-v0` | Word search Q&A |
| `Puzzles/Zuma-QA-v0` | Zuma Q&A |

---

### Multi-Turn Environments (74)

Multi-turn environments involve sustained interaction over multiple steps.

#### Games (31)

Classic board, card, and puzzle games.

| Environment ID | Description |
|----------------|-------------|
| `Games/Alquerque-v0` | Alquerque (Spanish checkers variant) |
| `Games/Breakthrough-v0` | Breakthrough board game |
| `Games/Chess-v0` | Chess |
| `Games/ConnectFour-v0` | Connect Four (single-agent vs built-in opponent) |
| `Games/ConnectFourMultiAgent-v0` | Connect Four (multi-agent) |
| `Games/Crosswords-v0` | Crossword puzzle |
| `Games/Crusade-v0` | Crusade board game |
| `Games/FifteenPuzzle-v0` | 15-puzzle sliding tiles |
| `Games/FrozenLake-v0` | Frozen lake navigation |
| `Games/Game2048-v0` | 2048 number merging game |
| `Games/GinRummy-v0` | Gin Rummy card game |
| `Games/Go-v0` | Go (Weiqi) |
| `Games/LeducHoldem-v0` | Leduc Hold'em poker |
| `Games/LightsOut-v0` | Lights Out puzzle |
| `Games/LinesOfAction-v0` | Lines of Action board game |
| `Games/Minesweeper-v0` | Minesweeper |
| `Games/Nim-v0` | Nim |
| `Games/Othello-v0` | Othello (Reversi) |
| `Games/PegJump-v0` | Peg solitaire |
| `Games/RushHour-v0` | Rush Hour sliding block puzzle |
| `Games/SimpleTak-v0` | Simplified Tak board game |
| `Games/Sokoban-v0` | Sokoban box-pushing puzzle |
| `Games/Sudoku-v0` | 9×9 Sudoku |
| `Games/TexasHoldem-v0` | Texas Hold'em poker |
| `Games/TexasHoldemNoLimit-v0` | No-Limit Texas Hold'em |
| `Games/TicTacToe-v0` | Tic-Tac-Toe |
| `Games/TowerOfHanoiMultiTurn-v0` | Tower of Hanoi (multi-turn) |
| `Games/UltimateTicTacToe-v0` | Ultimate Tic-Tac-Toe |
| `Games/WildTicTacToe-v0` | Wild Tic-Tac-Toe |
| `Games/WordSearch-v0` | Word search |
| `Games/Wordle-v0` | Wordle |

#### Spatial (30)

2D/3D spatial navigation and object interaction tasks.

| Environment ID | Description |
|----------------|-------------|
| `Spatial/CollectHealth-v0` | Collect health items in 3D space |
| `Spatial/DoorKey-v0` | Find key and unlock door in a 2D grid |
| `Spatial/DynamicObstacles-v0` | Navigate around dynamic obstacles in 2D |
| `Spatial/Empty-v0` | Empty room navigation baseline (2D) |
| `Spatial/FourRooms2D-v0` | Four-room navigation (2D) |
| `Spatial/FourRooms3D-v0` | Four-room navigation (3D) |
| `Spatial/Hallway-v0` | 3D hallway navigation |
| `Spatial/LavaGap-v0` | Cross a lava gap (2D) |
| `Spatial/Maze-v0` | 3D maze navigation |
| `Spatial/MazeS2-v0` | 3D maze (medium) |
| `Spatial/MazeS3-v0` | 3D maze (large) |
| `Spatial/MazeS3Fast-v0` | 3D maze (large, fast mode) |
| `Spatial/MultiRoom-v0` | Multi-room navigation (2D) |
| `Spatial/OneRoom-v0` | Single-room navigation (3D) |
| `Spatial/OneRoomS6-v0` | Single-room navigation, large (3D) |
| `Spatial/OneRoomS6Fast-v0` | Single-room navigation, large, fast (3D) |
| `Spatial/PickupObjects-v0` | Pick up objects in 3D space |
| `Spatial/PutNext-v0` | Place objects at target locations (3D) |
| `Spatial/RoomObjects-v0` | Multi-room object interaction (3D) |
| `Spatial/Sidewalk-v0` | 3D sidewalk navigation |
| `Spatial/Sign-v0` | Navigate by signs (3D) |
| `Spatial/TMaze-v0` | T-maze navigation (3D) |
| `Spatial/TMazeLeft-v0` | T-maze, left turn (3D) |
| `Spatial/TMazeRight-v0` | T-maze, right turn (3D) |
| `Spatial/ThreeRooms-v0` | Three-room navigation (3D) |
| `Spatial/Unlock-v0` | Unlock a room (2D) |
| `Spatial/WallGap-v0` | Navigate through wall gaps (3D) |
| `Spatial/YMaze-v0` | Y-maze navigation (3D) |
| `Spatial/YMazeLeft-v0` | Y-maze, left turn (3D) |
| `Spatial/YMazeRight-v0` | Y-maze, right turn (3D) |

#### Temporal (13)

Classic Sega Genesis retro games via [stable-retro](https://github.com/Farama-Foundation/stable-retro). See [Temporal README](gym_v/envs/multi_turn/temporal/README.md) for ROM setup.

| Environment ID | Description |
|----------------|-------------|
| `Temporal/Airstriker-v0` | Airstriker (free homebrew) |
| `Temporal/AlteredBeast-v0` | Altered Beast |
| `Temporal/CastleOfIllusion-v0` | Castle of Illusion |
| `Temporal/CastlevaniaBloodlines-v0` | Castlevania: Bloodlines |
| `Temporal/Columns-v0` | Columns |
| `Temporal/DynamiteHeaddy-v0` | Dynamite Headdy |
| `Temporal/GoldenAxe-v0` | Golden Axe |
| `Temporal/KidChameleon-v0` | Kid Chameleon |
| `Temporal/MortalKombatII-v0` | Mortal Kombat II |
| `Temporal/SpaceHarrierII-v0` | Space Harrier II |
| `Temporal/StreetsOfRage2-v0` | Streets of Rage 2 |
| `Temporal/Strider-v0` | Strider |
| `Temporal/ThunderForceIII-v0` | Thunder Force III |

---

## Key Findings

Using Gym-V, our experiments reveal several insights for training vision-language agents:

1. **Observation scaffolding > RL algorithm choice.** Captions, game rules, and interaction history determine whether learning succeeds at all — more so than the choice between GRPO, GSPO, or SAPO.

2. **Diverse training generalizes; narrow training hurts.** Cross-domain curricula transfer broadly, while training on a single domain can cause negative transfer. Multi-turn interaction amplifies both effects.

3. **RL closes the gap.** A 7B model trained with RL on Gym-V environments can surpass much larger models' zero-shot performance on several task categories.

For full results, see our paper.

## License

This project is for research use. See [LICENSE](LICENSE) for details.

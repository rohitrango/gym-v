"""GameRL multi-turn (interactive) environments.

These envs are step-based games where an agent takes multiple actions per episode.
The single-turn Q&A variants live under `gym_v.envs.gamerl`.
"""

from gym_v.envs.gamerl_multiturn.langton_ant import GameRLLangtonAntEnv
from gym_v.envs.gamerl_multiturn.lifegame import GameRLLifegameEnv
from gym_v.envs.gamerl_multiturn.maze import GameRLMazeEnv
from gym_v.envs.gamerl_multiturn.minesweeper import GameRLMinesweeperEnv
from gym_v.envs.gamerl_multiturn.pacman import GameRLPacmanEnv
from gym_v.envs.gamerl_multiturn.snake import GameRLSnakeEnv
from gym_v.envs.gamerl_multiturn.space_invaders import GameRLSpaceInvadersEnv
from gym_v.envs.gamerl_multiturn.sudoku import GameRLSudokuEnv
from gym_v.envs.gamerl_multiturn.tetris import GameRLTetrisEnv

# Game-RL Q&A environments (single-turn)
from gym_v.envs.gamerl.chess_ranger_qa import GameRLChessRangerQAEnv
from gym_v.envs.gamerl.freecell_qa import GameRLFreecellQAEnv
from gym_v.envs.gamerl.hue_qa import GameRLHueQAEnv
from gym_v.envs.gamerl.jewel2_qa import GameRLJewel2QAEnv
from gym_v.envs.gamerl.klondike_qa import GameRLKlondikeQAEnv
from gym_v.envs.gamerl.langton_ant_qa import GameRLLangtonAntQAEnv
from gym_v.envs.gamerl.lifegame_qa import GameRLLifegameQAEnv
from gym_v.envs.gamerl.maze_3d_qa import GameRL3dMazeQAEnv
from gym_v.envs.gamerl.maze_qa import GameRLMazeQAEnv
from gym_v.envs.gamerl.minecraft_qa import GameRLMinecraftQAEnv
from gym_v.envs.gamerl.minesweeper_qa import GameRLMinesweeperQAEnv
from gym_v.envs.gamerl.pacman_qa import GameRLPacmanQAEnv
from gym_v.envs.gamerl.pyramidchess_qa import GameRLPyramidChessQAEnv
from gym_v.envs.gamerl.rhythm_game_qa import GameRLRhythmGameQAEnv
from gym_v.envs.gamerl.rubiks_cube_qa import GameRLRubiksCubeQAEnv
from gym_v.envs.gamerl.snake_qa import GameRLSnakeQAEnv
from gym_v.envs.gamerl.sokoban_qa import GameRLSokobanQAEnv
from gym_v.envs.gamerl.space_invaders_qa import GameRLSpaceInvadersQAEnv
from gym_v.envs.gamerl.spider_solitaire_qa import GameRLSpiderSolitaireQAEnv
from gym_v.envs.gamerl.star_battle_qa import GameRLStarBattleQAEnv
from gym_v.envs.gamerl.sudoku_qa import GameRLSudokuQAEnv
from gym_v.envs.gamerl.tangram_qa import GameRLTangramQAEnv
from gym_v.envs.gamerl.tents_qa import GameRLTentsQAEnv
from gym_v.envs.gamerl.tetris_qa import GameRLTetrisQAEnv
from gym_v.envs.gamerl.threed_reconstruction_qa import GameRL3DReconstructionQAEnv
from gym_v.envs.gamerl.tictactoe_qa import GameRLTicTacToeQAEnv
from gym_v.envs.gamerl.turing_machine_2d_qa import GameRL2dTuringMachineQAEnv
from gym_v.envs.gamerl.ultra_tictactoe_qa import GameRLUltraTicTacToeQAEnv
from gym_v.envs.gamerl.word_search_qa import GameRLWordSearchQAEnv
from gym_v.envs.gamerl.zuma_qa import GameRLZumaQAEnv

# Multi-turn (interactive) Game-RL envs live in `gym_v.envs.gamerl_multiturn`.
# Re-export them here for convenience.
from gym_v.envs.gamerl_multiturn.langton_ant import GameRLLangtonAntEnv
from gym_v.envs.gamerl_multiturn.lifegame import GameRLLifegameEnv
from gym_v.envs.gamerl_multiturn.maze import GameRLMazeEnv
from gym_v.envs.gamerl_multiturn.minesweeper import GameRLMinesweeperEnv
from gym_v.envs.gamerl_multiturn.pacman import GameRLPacmanEnv
from gym_v.envs.gamerl_multiturn.snake import GameRLSnakeEnv
from gym_v.envs.gamerl_multiturn.space_invaders import GameRLSpaceInvadersEnv
from gym_v.envs.gamerl_multiturn.sudoku import GameRLSudokuEnv
from gym_v.envs.gamerl_multiturn.tetris import GameRLTetrisEnv

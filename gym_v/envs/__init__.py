"""Registers the internal gym-v envs then loads the env plugins for module using the entry point."""

from gym_v.envs.registration import register

# ReasoningGym environments
register(
    id="ReasoningGym/Sudoku-v0",
    entry_point="gym_v.envs.reasongym.sudoku:ReasoningGymSudokuEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=64,
        padding=24,
    ),
)

register(
    id="ReasoningGym/Maze-v0",
    entry_point="gym_v.envs.reasongym.maze:ReasoningGymMazeEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=48,
        padding=24,
    ),
)

register(
    id="ReasoningGym/NQueens-v0",
    entry_point="gym_v.envs.reasongym.n_queens:ReasoningGymNQueensEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=64,
        padding=24,
    ),
)

register(
    id="ReasoningGym/GameOfLife-v0",
    entry_point="gym_v.envs.reasongym.game_of_life:ReasoningGymGameOfLifeEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=32,
        padding=16,
    ),
)

register(
    id="ReasoningGym/TowerOfHanoi-v0",
    entry_point="gym_v.envs.reasongym.tower_of_hanoi:ReasoningGymTowerOfHanoiEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        peg_width=150,
        peg_height=250,
        padding=40,
    ),
)

register(
    id="ReasoningGym/KnightSwap-v0",
    entry_point="gym_v.envs.reasongym.knight_swap:ReasoningGymKnightSwapEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=64,
        padding=24,
    ),
)

register(
    id="ReasoningGym/MiniSudoku-v0",
    entry_point="gym_v.envs.reasongym.mini_sudoku:ReasoningGymMiniSudokuEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=80,
        padding=24,
    ),
)

register(
    id="ReasoningGym/Survo-v0",
    entry_point="gym_v.envs.reasongym.survo:ReasoningGymSurvoEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=64,
        padding=24,
    ),
)

register(
    id="ReasoningGym/Kakurasu-v0",
    entry_point="gym_v.envs.reasongym.kakurasu:ReasoningGymKakurasuEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=56,
        padding=40,
    ),
)

register(
    id="ReasoningGym/Tsumego-v0",
    entry_point="gym_v.envs.reasongym.tsumego:ReasoningGymTsumegoEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=36,
        padding=24,
    ),
)

register(
    id="ReasoningGym/SpiralMatrix-v0",
    entry_point="gym_v.envs.reasongym.spiral_matrix:ReasoningGymSpiralMatrixEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=48,
        padding=24,
    ),
)

register(
    id="ReasoningGym/RotateMatrix-v0",
    entry_point="gym_v.envs.reasongym.rotate_matrix:ReasoningGymRotateMatrixEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=48,
        padding=24,
    ),
)

register(
    id="ReasoningGym/BinaryMatrix-v0",
    entry_point="gym_v.envs.reasongym.binary_matrix:ReasoningGymBinaryMatrixEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=40,
        padding=20,
    ),
)

register(
    id="ReasoningGym/LargestIsland-v0",
    entry_point="gym_v.envs.reasongym.largest_island:ReasoningGymLargestIslandEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=40,
        padding=24,
    ),
)

register(
    id="ReasoningGym/RottenOranges-v0",
    entry_point="gym_v.envs.reasongym.rotten_oranges:ReasoningGymRottenOrangesEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=36,
        padding=20,
    ),
)

register(
    id="ReasoningGym/ShortestPath-v0",
    entry_point="gym_v.envs.reasongym.shortest_path:ReasoningGymShortestPathEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=48,
        padding=24,
    ),
)

register(
    id="ReasoningGym/RectangleCount-v0",
    entry_point="gym_v.envs.reasongym.rectangle_count:ReasoningGymRectangleCountEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=8,
        padding=16,
    ),
)

register(
    id="ReasoningGym/CircuitLogic-v0",
    entry_point="gym_v.envs.reasongym.circuit_logic:ReasoningGymCircuitLogicEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        padding=24,
    ),
)

register(
    id="ReasoningGym/Arc1D-v0",
    entry_point="gym_v.envs.reasongym.arc_1d:ReasoningGymArc1DEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        cell_px=28,
        padding=24,
    ),
)

# TextArena environments
register(
    id="TextArena/Crosswords-v0",
    entry_point="gym_v.envs.textarena.crosswords:TextArenaCrosswordsEnv",
    max_episode_steps=100,
    kwargs=dict(
        hardcore=False,
        num_words=5,
        cell_size=48,
    ),
)

register(
    id="TextArena/FifteenPuzzle-v0",
    entry_point="gym_v.envs.textarena.fifteenpuzzle:TextArenaFifteenPuzzleEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=80,
    ),
)

register(
    id="TextArena/FrozenLake-v0",
    entry_point="gym_v.envs.textarena.frozenlake:TextArenaFrozenLakeEnv",
    max_episode_steps=100,
    kwargs=dict(
        size=4,
        num_holes=3,
        randomize_start_goal=False,
        tile_size=64,
    ),
)

register(
    id="TextArena/Game2048-v0",
    entry_point="gym_v.envs.textarena.game2048:TextArenaGame2048Env",
    max_episode_steps=100,
    kwargs=dict(
        target_tile=2048,
        tile_size=100,
    ),
)

register(
    id="TextArena/LightsOut-v0",
    entry_point="gym_v.envs.textarena.lightsout:TextArenaLightsOutEnv",
    max_episode_steps=100,
    kwargs=dict(
        size=5,
        cell_size=80,
    ),
)

register(
    id="TextArena/Minesweeper-v0",
    entry_point="gym_v.envs.textarena.minesweeper:TextArenaMinesweeperEnv",
    max_episode_steps=100,
    kwargs=dict(
        rows=8,
        cols=8,
        num_mines=10,
        cell_size=64,
    ),
)

register(
    id="TextArena/PegJump-v0",
    entry_point="gym_v.envs.textarena.pegjump:TextArenaPegJumpEnv",
    max_episode_steps=100,
    kwargs=dict(
        initial_empty=1,
        peg_size=80,
    ),
)

register(
    id="TextArena/RushHour-v0",
    entry_point="gym_v.envs.textarena.rushhour:TextArenaRushHourEnv",
    max_episode_steps=100,
    kwargs=dict(
        difficulty="easy",
        cell_size=80,
    ),
)

register(
    id="TextArena/Sokoban-v0",
    entry_point="gym_v.envs.textarena.sokoban:TextArenaSokobanEnv",
    max_episode_steps=100,
    kwargs=dict(
        dim_room=(6, 6),
        num_boxes=3,
        tile_size=48,
    ),
)

register(
    id="TextArena/Sudoku-v0",
    entry_point="gym_v.envs.textarena.sudoku:TextArenaSudokuEnv",
    max_episode_steps=100,
    kwargs=dict(
        clues=30,
        cell_size=50,
    ),
)

register(
    id="TextArena/TowerOfHanoi-v0",
    entry_point="gym_v.envs.textarena.towerofhanoi:TextArenaTowerOfHanoiEnv",
    max_episode_steps=100,
    kwargs=dict(
        num_disks=3,
    ),
)

register(
    id="TextArena/Wordle-v0",
    entry_point="gym_v.envs.textarena.wordle:TextArenaWordleEnv",
    max_episode_steps=100,
    kwargs=dict(
        word_length=5,
        num_guesses=6,
        hardcore=False,
        cell_size=60,
    ),
)

register(
    id="TextArena/WordSearch-v0",
    entry_point="gym_v.envs.textarena.wordsearch:TextArenaWordSearchEnv",
    max_episode_steps=100,
    kwargs=dict(
        hardcore=False,
        cell_size=60,
    ),
)

# Game-RL environments
register(
    id="GameRL/Snake-v0",
    entry_point="gym_v.envs.gamerl.snake:GameRLSnakeEnv",
    max_episode_steps=200,
    kwargs=dict(
        width=10,
        height=10,
        initial_snake_length=3,
        cell_size=40,
    ),
)

# Offline datasets
register(
    id="Offline/SingleTurn-v0",
    entry_point="gym_v.envs.offline.single_turn:OfflineSingleTurnEnv",
    max_episode_steps=1,
    kwargs=dict(
        grader="exact_match",
        description=None,
        shuffle=True,
    ),
)

# VGRP-Bench environments
register(
    id="VGRP/Binairo-v0",
    entry_point="gym_v.envs.vgrp.binairo:VGRPBinairoEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=6,
        num_hints=12,
        cell_px=60,
        padding=24,
    ),
)

register(
    id="VGRP/Thermometers-v0",
    entry_point="gym_v.envs.vgrp.thermometers:VGRPThermometersEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=5,
        num_hints=0,
        cell_px=60,
        padding=50,
    ),
)

register(
    id="VGRP/TreesAndTents-v0",
    entry_point="gym_v.envs.vgrp.treesandtents:VGRPTreesAndTentsEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=5,
        num_hints=0,
        cell_px=60,
        padding=50,
    ),
)

register(
    id="VGRP/Battleships-v0",
    entry_point="gym_v.envs.vgrp.battleships:VGRPBattleshipsEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=6,
        num_hints=0,
        cell_px=55,
        padding=50,
    ),
)

register(
    id="VGRP/Renzoku-v0",
    entry_point="gym_v.envs.vgrp.renzoku:VGRPRenzokuEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=9,
        cell_px=50,
        padding=30,
    ),
)

register(
    id="VGRP/Futoshiki-v0",
    entry_point="gym_v.envs.vgrp.futoshiki:VGRPFutoshikiEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=5,
        cell_px=60,
        padding=30,
    ),
)

register(
    id="VGRP/Hitori-v0",
    entry_point="gym_v.envs.vgrp.hitori:VGRPHitoriEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=6,
        cell_px=50,
        padding=20,
    ),
)

register(
    id="VGRP/StarBattle-v0",
    entry_point="gym_v.envs.vgrp.starbattle:VGRPStarBattleEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=8,
        stars_per_group=1,
        cell_px=50,
        padding=20,
    ),
)

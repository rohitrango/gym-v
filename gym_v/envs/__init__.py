"""Registers the internal gym-v envs then loads the env plugins for module using the entry point."""

from gym_v.envs.registration import register

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

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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
    ),
)

register(
    id="ReasoningGym/CircuitLogic-v0",
    entry_point="gym_v.envs.reasongym.circuit_logic:ReasoningGymCircuitLogicEnv",
    max_episode_steps=1,
    kwargs=dict(
        dataset_kwargs=dict(size=500),
        padding=24,
        num_players=1,
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
        num_players=1,
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
        num_players=1,
    ),
)

register(
    id="TextArena/FifteenPuzzle-v0",
    entry_point="gym_v.envs.textarena.fifteenpuzzle:TextArenaFifteenPuzzleEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=80,
        num_players=1,
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
        num_players=1,
    ),
)

register(
    id="TextArena/Game2048-v0",
    entry_point="gym_v.envs.textarena.game2048:TextArenaGame2048Env",
    max_episode_steps=100,
    kwargs=dict(
        target_tile=2048,
        tile_size=100,
        num_players=1,
    ),
)

register(
    id="TextArena/LightsOut-v0",
    entry_point="gym_v.envs.textarena.lightsout:TextArenaLightsOutEnv",
    max_episode_steps=100,
    kwargs=dict(
        size=5,
        cell_size=80,
        num_players=1,
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
        num_players=1,
    ),
)

register(
    id="TextArena/PegJump-v0",
    entry_point="gym_v.envs.textarena.pegjump:TextArenaPegJumpEnv",
    max_episode_steps=100,
    kwargs=dict(
        initial_empty=1,
        peg_size=80,
        num_players=1,
    ),
)

register(
    id="TextArena/RushHour-v0",
    entry_point="gym_v.envs.textarena.rushhour:TextArenaRushHourEnv",
    max_episode_steps=100,
    kwargs=dict(
        difficulty="easy",
        cell_size=80,
        num_players=1,
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
        num_players=1,
    ),
)

register(
    id="TextArena/Sudoku-v0",
    entry_point="gym_v.envs.textarena.sudoku:TextArenaSudokuEnv",
    max_episode_steps=100,
    kwargs=dict(
        clues=30,
        cell_size=50,
        num_players=1,
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
    id="TextArena/Othello-v0",
    entry_point="gym_v.envs.multi_players.textarena.othello:TextArenaOthello",
    max_episode_steps=100,
    kwargs=dict(
        board_size=8,
        show_valid=True,
        tile_size=80,
        num_players=2,
    ),
)

register(
    id="TextArena/ConnectFour-v0",
    entry_point="gym_v.envs.multi_players.textarena.connectfour:TextArenaConnectFour",
    max_episode_steps=100,
    kwargs=dict(
        num_rows=6,
        num_cols=7,
        tile_size=80,
        num_players=2,
    ),
)

register(
    id="TextArena/Nim-v0",
    entry_point="gym_v.envs.multi_players.textarena.nim:TextArenaNim",
    max_episode_steps=50,
    kwargs=dict(
        piles=[3, 4, 5],
        pile_width=100,
        num_players=2,
    ),
)

register(
    id="TextArena/WildTicTacToe-v0",
    entry_point="gym_v.envs.multi_players.textarena.wildtictactoe:TextArenaWildTicTacToe",
    max_episode_steps=20,
    kwargs=dict(
        tile_size=120,
        num_players=2,
    ),
)

register(
    id="TextArena/Breakthrough-v0",
    entry_point="gym_v.envs.multi_players.textarena.breakthrough:TextArenaBreakthrough",
    max_episode_steps=200,
    kwargs=dict(
        board_size=8,
        tile_size=60,
        num_players=2,
    ),
)

register(
    id="TextArena/Alquerque-v0",
    entry_point="gym_v.envs.multi_players.textarena.alquerque:TextArenaAlquerque",
    max_episode_steps=200,
    kwargs=dict(
        tile_size=80,
        num_players=2,
    ),
)

register(
    id="TextArena/LinesOfAction-v0",
    entry_point="gym_v.envs.multi_players.textarena.linesofaction:TextArenaLinesOfAction",
    max_episode_steps=200,
    kwargs=dict(
        tile_size=60,
        num_players=2,
    ),
)

register(
    id="TextArena/SimpleTak-v0",
    entry_point="gym_v.envs.multi_players.textarena.simpletak:TextArenaSimpleTak",
    max_episode_steps=50,
    kwargs=dict(
        board_size=5,
        cell_size=80,
        num_players=2,
    ),
)

register(
    id="TextArena/Crusade-v0",
    entry_point="gym_v.envs.multi_players.textarena.crusade:TextArenaCrusade",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=60,
        num_players=2,
    ),
)

register(
    id="TextArena/UltimateTicTacToe-v0",
    entry_point="gym_v.envs.multi_players.textarena.ultimatetictactoe:TextArenaUltimateTicTacToe",
    max_episode_steps=100,
    kwargs=dict(
        mini_board_size=200,
        num_players=2,
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
        num_players=1,
    ),
)

register(
    id="TextArena/WordSearch-v0",
    entry_point="gym_v.envs.textarena.wordsearch:TextArenaWordSearchEnv",
    max_episode_steps=100,
    kwargs=dict(
        hardcore=False,
        cell_size=60,
        num_players=1,
    ),
)

# Minigrid environments
register(
    id="Minigrid/Empty-v0",
    entry_point="gym_v.envs.minigrid.empty:MinigridEmptyEnv",
    max_episode_steps=256,
    kwargs=dict(
        size=8,
        agent_start_pos=(1, 1),
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Minigrid/DoorKey-v0",
    entry_point="gym_v.envs.minigrid.doorkey:MinigridDoorKeyEnv",
    max_episode_steps=640,
    kwargs=dict(
        size=8,
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Minigrid/FourRooms-v0",
    entry_point="gym_v.envs.minigrid.fourrooms:MinigridFourRoomsEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Minigrid/Unlock-v0",
    entry_point="gym_v.envs.minigrid.unlock:MinigridUnlockEnv",
    max_episode_steps=100,
    kwargs=dict(
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Minigrid/MultiRoom-v0",
    entry_point="gym_v.envs.minigrid.multiroom:MinigridMultiRoomEnv",
    max_episode_steps=100,
    kwargs=dict(
        min_num_rooms=6,
        max_num_rooms=6,
        max_room_size=10,
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Minigrid/DynamicObstacles-v0",
    entry_point="gym_v.envs.minigrid.dynamicobstacles:MinigridDynamicObstaclesEnv",
    max_episode_steps=256,
    kwargs=dict(
        size=8,
        n_obstacles=4,
        tile_size=32,
        num_players=1,
    ),
)

register(
    id="Minigrid/LavaGap-v0",
    entry_point="gym_v.envs.minigrid.lavagap:MinigridLavaGapEnv",
    max_episode_steps=100,
    kwargs=dict(
        size=7,
        tile_size=32,
        num_players=1,
    ),
)

# Game-RL environments
register(
    id="GameRL/Snake-v0",
    entry_point="gym_v.envs.gamerl_multiturn.snake:GameRLSnakeEnv",
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
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
        num_players=1,
    ),
)
register(
    id="GameRL/Pacman-v0",
    entry_point="gym_v.envs.gamerl_multiturn.pacman:GameRLPacmanEnv",
    max_episode_steps=500,
    kwargs=dict(
        grid_size=16,
        wall_ratio=0.1,
        cell_size=25,
        num_players=1,
    ),
)

register(
    id="GameRL/Tetris-v0",
    entry_point="gym_v.envs.gamerl_multiturn.tetris:GameRLTetrisEnv",
    max_episode_steps=1000,
    kwargs=dict(
        rows=12,
        cols=8,
        cell_size=30,
        num_players=1,
    ),
)

register(
    id="GameRL/SpaceInvaders-v0",
    entry_point="gym_v.envs.gamerl_multiturn.space_invaders:GameRLSpaceInvadersEnv",
    max_episode_steps=200,
    kwargs=dict(
        enemy_rows=4,
        enemy_cols=6,
        enemy_area_rows=8,
        cell_width=50,
        cell_height=40,
        num_players=1,
    ),
)

register(
    id="GameRL/Maze-v0",
    entry_point="gym_v.envs.gamerl_multiturn.maze:GameRLMazeEnv",
    max_episode_steps=200,
    kwargs=dict(
        size="small",
        cell_size=40,
        num_players=1,
    ),
)

register(
    id="GameRL/Lifegame-v0",
    entry_point="gym_v.envs.gamerl_multiturn.lifegame:GameRLLifegameEnv",
    max_episode_steps=1000,
    kwargs=dict(
        grid_size=30,
        cell_size=20,
        random_init=True,
        init_density=0.3,
        num_players=1,
    ),
)

register(
    id="GameRL/LangtonAnt-v0",
    entry_point="gym_v.envs.gamerl_multiturn.langton_ant:GameRLLangtonAntEnv",
    max_episode_steps=1000,
    kwargs=dict(
        grid_size=15,
        cell_size=30,
        init_black_ratio=0.1,
        num_players=1,
    ),
)

register(
    id="RLVE/HitoriPuzzle-v0",
    entry_point="gym_v.envs.rlve.hitori_puzzle:RLVEHitoriPuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=4,
        cell_px=56,
        padding=24,
        num_players=1,
    ),
)

register(
    id="RLVE/SkyscraperPuzzle-v0",
    entry_point="gym_v.envs.rlve.skyscraper_puzzle:RLVESkyscraperPuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        n=3,
        cell_px=52,
        padding=28,
        num_players=1,
    ),
)

register(
    id="RLVE/LightUpPuzzle-v0",
    entry_point="gym_v.envs.rlve.light_up_puzzle:RLVELightUpPuzzleEnv",
    max_episode_steps=1,
    kwargs=dict(
        max_n_m=3,
        density_list=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        black_cell_density_range=(0.6, 0.95),
        cell_px=48,
        padding=24,
        num_players=1,
    ),
)

# Game-RL Q&A environments (single-turn)
register(
    id="GameRL/Snake-QA-v0",
    entry_point="gym_v.envs.gamerl.snake:GameRLSnakeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        width=10,
        height=10,
        initial_snake_length=(10, 20),
        cell_size=40,
        num_players=1,
    ),
)

register(
    id="GameRL/Maze-QA-v0",
    entry_point="gym_v.envs.gamerl.maze:GameRLMazeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        size="small",
        cell_size=40,
        num_players=1,
    ),
)

register(
    id="GameRL/Maze3D-QA-v0",
    entry_point="gym_v.envs.gamerl.maze_3d:GameRL3dMazeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=(8, 8, 7),
        num_players=1,
    ),
)

register(
    id="GameRL/Lifegame-QA-v0",
    entry_point="gym_v.envs.gamerl.lifegame:GameRLLifegameQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        cell_size=30,
        num_players=1,
    ),
)

register(
    id="GameRL/Freecell-QA-v0",
    entry_point="gym_v.envs.gamerl.freecell:GameRLFreecellQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        cascade_number=None,
        num_players=1,
    ),
)

register(
    id="GameRL/Hue-QA-v0",
    entry_point="gym_v.envs.gamerl.hue:GameRLHueQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        board_size=None,
        num_lines=None,
        cell_size=60,
        num_players=1,
    ),
)

register(
    id="GameRL/Jewel2-QA-v0",
    entry_point="gym_v.envs.gamerl.jewel2:GameRLJewel2QAEnv",
    max_episode_steps=1,
    kwargs=dict(
        size=5,
        question_type=None,
        num_players=1,
    ),
)

register(
    id="GameRL/LangtonAnt-QA-v0",
    entry_point="gym_v.envs.gamerl.langton_ant:GameRLLangtonAntQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        cell_size=30,
        num_players=1,
    ),
)

register(
    id="GameRL/Minesweeper-QA-v0",
    entry_point="gym_v.envs.gamerl.minesweeper:GameRLMinesweeperQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        difficulty=None,
        cell_size=60,
        num_players=1,
    ),
)

register(
    id="GameRL/Sudoku-QA-v0",
    entry_point="gym_v.envs.gamerl.sudoku:GameRLSudokuQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        size=None,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="GameRL/Pacman-QA-v0",
    entry_point="gym_v.envs.gamerl.pacman:GameRLPacmanQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=16,
        wall_ratio=0.1,
        cell_size=25,
        num_players=1,
    ),
)

register(
    id="GameRL/RhythmGame-QA-v0",
    entry_point="gym_v.envs.gamerl.rhythm_game:GameRLRhythmGameQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        difficulty=None,
        cell_size=40,
        num_players=1,
    ),
)

register(
    id="GameRL/RubiksCube-QA-v0",
    entry_point="gym_v.envs.gamerl.rubiks_cube:GameRLRubiksCubeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        num_moves=None,
        num_players=1,
    ),
)

register(
    id="GameRL/Sokoban-QA-v0",
    entry_point="gym_v.envs.gamerl.sokoban:GameRLSokobanQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        size=5,
        num_boxes=1,
        num_players=1,
    ),
)

register(
    id="GameRL/SpiderSolitaire-QA-v0",
    entry_point="gym_v.envs.gamerl.spider_solitaire:GameRLSpiderSolitaireQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        num_waste=10,
        circular=False,
        open=False,
        question_type=None,
        num_players=1,
    ),
)

register(
    id="GameRL/Tangram-QA-v0",
    entry_point="gym_v.envs.gamerl.tangram:GameRLTangramQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        num_seeds=None,
        num_pieces_to_remove=None,
        num_players=1,
    ),
)

register(
    id="GameRL/Tetris-QA-v0",
    entry_point="gym_v.envs.gamerl.tetris:GameRLTetrisQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        rows=12,
        cols=8,
        cell_size=30,
        num_players=1,
    ),
)

register(
    id="GameRL/TicTacToe-QA-v0",
    entry_point="gym_v.envs.gamerl.tictactoe:GameRLTicTacToeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        num_players=1,
    ),
)

register(
    id="GameRL/UltraTicTacToe-QA-v0",
    entry_point="gym_v.envs.gamerl.ultra_tictactoe:GameRLUltraTicTacToeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        plot_level="Easy",
        question_type=None,
        num_players=1,
    ),
)

register(
    id="GameRL/TuringMachine2d-QA-v0",
    entry_point="gym_v.envs.gamerl.turing_machine_2d:GameRL2dTuringMachineQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        num_states=2,
        num_symbols=2,
        max_steps=8,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="GameRL/Tents-QA-v0",
    entry_point="gym_v.envs.gamerl.tents:GameRLTentsQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        num_trees=None,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="GameRL/SpaceInvaders-QA-v0",
    entry_point="gym_v.envs.gamerl.space_invaders:GameRLSpaceInvadersQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        enemy_rows=4,
        enemy_cols=6,
        enemy_area_rows=8,
        cell_width=50,
        cell_height=40,
        num_players=1,
    ),
)

register(
    id="GameRL/StarBattle-QA-v0",
    entry_point="gym_v.envs.gamerl.star_battle:GameRLStarBattleQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=6,
        stars_per_region=1,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="GameRL/WordSearch-QA-v0",
    entry_point="gym_v.envs.gamerl.word_search:GameRLWordSearchQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        grid_size=None,
        cell_size=50,
        num_players=1,
    ),
)

register(
    id="GameRL/Zuma-QA-v0",
    entry_point="gym_v.envs.gamerl.zuma:GameRLZumaQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        curve_type=None,
        num_balls=None,
        ball_radius=0.3,
        num_players=1,
    ),
)

register(
    id="GameRL/3DReconstruction-QA-v0",
    entry_point="gym_v.envs.gamerl.threed_reconstruction:GameRL3DReconstructionQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        plot_level="Easy",
        question_type=None,
        num_players=1,
    ),
)

register(
    id="GameRL/ChessRanger-QA-v0",
    entry_point="gym_v.envs.gamerl.chess_ranger:GameRLChessRangerQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        num_pieces=6,
        question_type=None,
        num_players=1,
    ),
)

register(
    id="GameRL/PyramidChess-QA-v0",
    entry_point="gym_v.envs.gamerl.pyramidchess:GameRLPyramidChessQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        plot_level="Easy",
        question_type=None,
        num_players=1,
    ),
)

register(
    id="GameRL/Klondike-QA-v0",
    entry_point="gym_v.envs.gamerl.klondike:GameRLKlondikeQAEnv",
    max_episode_steps=1,
    kwargs=dict(
        question_type=None,
        num_players=1,
    ),
)

# Sphinx environments (procedurally generated) - Grid style (ARC-like)
register(
    id="Sphinx/TransformResult-v0",
    entry_point="gym_v.envs.sphinx.transform_result:SphinxTransformResultEnv",
    max_episode_steps=1,
    kwargs=dict(
        grid_size=5,
        num_colors=4,
        cell_size=40,
        option_size=280,
        padding=20,
    ),
)

register(
    id="Sphinx/SymmetryFill-v0",
    entry_point="gym_v.envs.sphinx.symmetry_fill:SphinxSymmetryFillEnv",
    max_episode_steps=1,
    kwargs=dict(
        cell_grid_size=4,
        num_colors=3,
        cell_size=100,
        option_size=200,
        padding=15,
    ),
)

register(
    id="Sphinx/TransformResultPoly-v0",
    entry_point="gym_v.envs.sphinx.transform_result:SphinxTransformResultPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=300,
        num_points=8,
        line_width=3,
        grid_divisions=8,
        option_size=280,
        padding=20,
    ),
)

register(
    id="Sphinx/SymmetryFillPoly-v0",
    entry_point="gym_v.envs.sphinx.symmetry_fill:SphinxSymmetryFillPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        cell_size=200,
        line_width=4,
        option_size=200,
        padding=15,
    ),
)

register(
    id="Sphinx/OddOneOut-v0",
    entry_point="gym_v.envs.sphinx.odd_one_out:SphinxOddOneOutEnv",
    max_episode_steps=1,
    kwargs=dict(
        grid_size=5,
        num_colors=4,
        cell_size=40,
        option_size=200,
        padding=15,
    ),
)

register(
    id="Sphinx/OddOneOutPoly-v0",
    entry_point="gym_v.envs.sphinx.odd_one_out:SphinxOddOneOutPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=200,
        num_points=8,
        line_width=3,
        grid_divisions=8,
        option_size=200,
        padding=15,
    ),
)

register(
    id="Sphinx/SequenceCompletion-v0",
    entry_point="gym_v.envs.sphinx.sequence_completion:SphinxSequenceCompletionEnv",
    max_episode_steps=1,
    kwargs=dict(
        grid_size=4,
        num_colors=3,
        cell_size=35,
        sequence_length=4,
        option_size=150,
        padding=10,
    ),
)

register(
    id="Sphinx/SequenceCompletionPoly-v0",
    entry_point="gym_v.envs.sphinx.sequence_completion:SphinxSequenceCompletionPolyEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=200,
        num_points=6,
        line_width=3,
        grid_divisions=8,
        sequence_length=4,
        option_size=150,
        padding=10,
    ),
)

# Perception environments
register(
    id="Perception/ChartToTable-v0",
    entry_point="gym_v.envs.perception.chart_to_table:PerceptionChartToTableEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        max_categories=8,
        num_players=1,
    ),
)

# Graph Algorithm Perception environments
register(
    id="Perception/GraphToAdjacency-v0",
    entry_point="gym_v.envs.perception.graph_to_adjacency:PerceptionGraphToAdjacencyEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        min_nodes=4,
        max_nodes=8,
        num_players=1,
    ),
)

register(
    id="Perception/TreeToTraversal-v0",
    entry_point="gym_v.envs.perception.tree_to_traversal:PerceptionTreeToTraversalEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        min_nodes=5,
        max_nodes=12,
        num_players=1,
    ),
)

register(
    id="Perception/DAGToTopoOrder-v0",
    entry_point="gym_v.envs.perception.dag_to_topo_order:PerceptionDAGToTopoOrderEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        min_nodes=5,
        max_nodes=9,
        num_players=1,
    ),
)

register(
    id="Perception/GraphToMST-v0",
    entry_point="gym_v.envs.perception.graph_to_mst:PerceptionGraphToMSTEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        min_nodes=5,
        max_nodes=8,
        num_players=1,
    ),
)

register(
    id="Perception/FlowNetwork-v0",
    entry_point="gym_v.envs.perception.flow_network:PerceptionFlowNetworkEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        min_nodes=5,
        max_nodes=8,
        num_players=1,
    ),
)

# Mathematical Function Perception environments
register(
    id="Perception/FunctionGraph-v0",
    entry_point="gym_v.envs.perception.function_graph:PerceptionFunctionGraphEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        x_range=(-5, 5),
        num_players=1,
    ),
)

register(
    id="Perception/ContourPlot-v0",
    entry_point="gym_v.envs.perception.contour_plot:PerceptionContourPlotEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        xy_range=(-3, 3),
        num_players=1,
    ),
)

register(
    id="Perception/PolarPlot-v0",
    entry_point="gym_v.envs.perception.polar_plot:PerceptionPolarPlotEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        num_players=1,
    ),
)

register(
    id="Perception/VectorField-v0",
    entry_point="gym_v.envs.perception.vector_field:PerceptionVectorFieldEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        xy_range=(-3, 3),
        grid_density=15,
        num_players=1,
    ),
)

register(
    id="Perception/ParametricCurve-v0",
    entry_point="gym_v.envs.perception.parametric_curve:PerceptionParametricCurveEnv",
    max_episode_steps=1,
    kwargs=dict(
        img_size=(640, 480),
        num_players=1,
    ),
)

# Stable-Retro environments (optional dependency)
# These require stable-retro to be installed: uv sync --extra stable-retro
register(
    id="Retro/Airstriker-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="Airstriker-Genesis",
        num_players=1,
    ),
)

register(
    id="Retro/GoldenAxe-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="GoldenAxe-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/StreetsOfRage2-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="StreetsOfRage2-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/MortalKombatII-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="MortalKombatII-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/Strider-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="Strider-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/CastleOfIllusion-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="CastleOfIllusion-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/CastlevaniaBloodlines-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="CastlevaniaBloodlines-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/Columns-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="Columns-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/ThunderForceIII-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="ThunderForceIII-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/SpaceHarrierII-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="SpaceHarrierII-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/AlteredBeast-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="AlteredBeast-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/DynamiteHeaddy-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="DynamiteHeaddy-Genesis-v0",
        num_players=1,
    ),
)

register(
    id="Retro/KidChameleon-v0",
    entry_point="gym_v.envs.stable_retro:RetroGymVEnv",
    max_episode_steps=10000,
    kwargs=dict(
        game="KidChameleon-Genesis-v0",
        num_players=1,
    ),
)

# PettingZoo environments
register(
    id="PettingZoo/Chess-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.chess:PettingZooChess",
    max_episode_steps=500,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="PettingZoo/ConnectFour-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.connectfour:PettingZooConnectFour",
    max_episode_steps=100,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="PettingZoo/GinRummy-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.gin_rummy:PettingZooGinRummy",
    max_episode_steps=200,
    kwargs=dict(
        num_players=2,
        knock_reward=0.5,
        gin_reward=1.0,
        opponents_hand_visible=False,
    ),
)

register(
    id="PettingZoo/Go-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.go:PettingZooGo",
    max_episode_steps=1000,
    kwargs=dict(
        num_players=2,
        board_size=19,
        komi=7.5,
    ),
)

register(
    id="PettingZoo/LeducHoldem-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.leduc_holdem:PettingZooLeducHoldem",
    max_episode_steps=100,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="PettingZoo/TexasHoldem-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.texas_holdem:PettingZooTexasHoldem",
    max_episode_steps=200,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="PettingZoo/TexasHoldemNoLimit-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.texas_holdem_no_limit:PettingZooTexasHoldemNoLimit",
    max_episode_steps=200,
    kwargs=dict(
        num_players=2,
    ),
)

register(
    id="PettingZoo/TicTacToe-v0",
    entry_point="gym_v.envs.multi_players.pettingzoo.tictactoe:PettingZooTicTacToe",
    max_episode_steps=20,
    kwargs=dict(
        num_players=2,
    ),
)

# VLMEvalKit environment
register(
    id="VLMEval-Base-v0",
    entry_point="gym_v.envs.eval.vlmeval:VLMEvalEnv",
    max_episode_steps=1,
    kwargs={},
)

# GenEval T2I environment
register(
    id="GenEval-v0",
    entry_point="gym_v.envs.eval.t2ieval:GenevalEnv",
    max_episode_steps=1,
    kwargs=dict(),
)

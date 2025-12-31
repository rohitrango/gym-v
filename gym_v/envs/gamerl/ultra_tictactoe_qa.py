"""Ultra TicTacToe QA Environment

This module implements an Ultra TicTacToe question-answering environment for the gym-v framework.
The game features a 3x3 grid of 3x3 boards (9 boards total, 81 cells).

Key Rules:
- First player (X) starts in the middle Nine-grid (2, 2)
- Where you play determines which board your opponent must play next
- Players score points for creating "three in a row" within individual Nine-grids
- Game ends when all middle cells (2,2) of each Nine-grid are filled

The environment supports 7 question types across Easy/Medium/Hard difficulty levels.

Source: /mnt/petrelfs/gujiawei/jiawei/env-v/Game-RL/src/ultra_tictactoe/
"""

from __future__ import annotations

import random
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation

GAME_RULES = """Now I'll give you a picture, which shows a screenshot of Ultra TicTacToe. The introduction of Ultra TicTacToe is as follows:
1. Board and coordinate representation: In this game, the board is divided into 9 3*3 squares(called Nine-grids). At the same time, we use $(i, j, row, col)$ to represent the coordinates of a cell: $(i, j)$ represents the coordinates of the Nine-grid; $(row, col)$ represents the coordinate of the cell within the Nine-grid; $i, j, row, col$ all range from 1 to 3. Two players take turns placing pieces on the board to mark the cells on the board, with the first player using "X" and the second player using "O" (this is the same as traditional TicTacToe).
2. Rules for placing chess pieces: After the game starts, the first player places a chess piece in any cell in the Nine-grid in the middle (i.e., the Nine-grid (2, 2)). After that, the coordinates of each chess piece placed in the Nine-grid are the same as the coordinates of the Nine-grid in which the opponent's last chess piece was placed; for example, if the first player places a chess piece at the coordinates (2, 2, 3, 1) in the first step, then the second player needs to choose a chess piece in the nine-square grid (3, 1) in the second step.
3. Scoring rules: For each player, each "Straight" (i.e., three identical chess pieces connected in a line, such as in the same row, the same column, or a diagonal line) in each Nine-grid is counted as 1 point. More than 1 point can be counted in each Nine-grid.

Now I will give you a question about the game. Please extract information from the picture I give you, think carefully, reason, and answer:
"""


class UTTTGameGrid:
    """Ultra TicTacToe game state manager.

    Manages a 3x3 grid of 3x3 boards with piece placement and scoring logic.
    Coordinates are represented as (i, j, row, col) where:
    - (i, j): Nine-grid position (1-3)
    - (row, col): Cell position within Nine-grid (1-3)
    """

    def __init__(self, plot_level="Easy"):
        """Initialize game grid.

        Args:
            plot_level: Difficulty level determining game length
        """
        self.plot_level = plot_level
        self.piece_coord = {
            f"({s}, {t})": {"X": [], "O": []} for s in range(1, 4) for t in range(1, 4)
        }
        self.point_record = {
            "X": {f"({s}, {t})": 0 for s in range(1, 4) for t in range(1, 4)},
            "O": {f"({s}, {t})": 0 for s in range(1, 4) for t in range(1, 4)},
        }
        self.all_points = {"X": 0, "O": 0}
        self.break_step_dict = {
            "Easy": {"low": 10, "high": 34},
            "Medium": {"low": 35, "high": 59},
            "Hard": {"low": 60, "high": 81},
        }
        self.middle_cell_count = 0
        self.last_step = None
        self.best_next_step = None

    def coord_to_str(self, row: int, col: int) -> str:
        """Convert coordinate to string format."""
        return f"({row}, {col})"

    def check_coord_avail(self, i: int, j: int, row: int, col: int) -> bool:
        """Check if a coordinate is available for placement.

        Args:
            i, j: Nine-grid position (1-3)
            row, col: Cell position within Nine-grid (1-3)

        Returns:
            True if cell is empty and valid
        """
        if not (1 <= i <= 3 and 1 <= j <= 3):
            return False
        if not (1 <= row <= 3 and 1 <= col <= 3):
            return False

        nine_grid = self.coord_to_str(i, j)
        cell = self.coord_to_str(row, col)

        return (
            cell not in self.piece_coord[nine_grid]["X"]
            and cell not in self.piece_coord[nine_grid]["O"]
        )

    def draw_piece(self, i: int, j: int, row: int, col: int, piece: str):
        """Place a piece at the specified coordinate.

        Args:
            i, j: Nine-grid position
            row, col: Cell position
            piece: "X" or "O"
        """
        if not self.check_coord_avail(i, j, row, col):
            return

        nine_grid = self.coord_to_str(i, j)
        cell = self.coord_to_str(row, col)
        self.piece_coord[nine_grid][piece].append(cell)

        # Update middle cell counter
        if row == 2 and col == 2:
            self.middle_cell_count += 1

    def random_piece_in_nine_grid(self, i: int, j: int, piece: str) -> tuple[int, int]:
        """Place a random piece in the specified Nine-grid.

        Returns:
            (row, col) of placed piece, or (0, 0) if no space
        """
        coord_avail_list = [
            (x, y)
            for x in range(1, 4)
            for y in range(1, 4)
            if self.check_coord_avail(i, j, x, y)
        ]
        if not coord_avail_list:
            return 0, 0

        row, col = random.choice(coord_avail_list)
        self.draw_piece(i, j, row, col, piece)
        return row, col

    def find_best_piece_in_nine_grid(
        self, i: int, j: int, piece: str
    ) -> tuple[int, list]:
        """Find coordinates that give maximum point increase.

        Returns:
            (max_point_diff, list of (row, col) coordinates achieving it)
        """
        coord_avail_list = [
            (x, y)
            for x in range(1, 4)
            for y in range(1, 4)
            if self.check_coord_avail(i, j, x, y)
        ]
        if not coord_avail_list:
            return 0, []

        add_point_record = {k: [] for k in range(5)}
        max_point_diff = 0
        old_point = self.point_record[piece][self.coord_to_str(i, j)]

        for row, col in coord_avail_list:
            new_point, _, _, _ = self.check_point_in_nine_grid(i, j, row, col, piece)
            point_diff = new_point - old_point
            if point_diff > max_point_diff:
                max_point_diff = point_diff
            add_point_record[point_diff].append((row, col))

        return max_point_diff, add_point_record[max_point_diff]

    def check_point_in_nine_grid(
        self, i: int = 1, j: int = 1, row: int = 0, col: int = 0, piece: str = "X"
    ) -> tuple[int, int, int, int]:
        """Calculate points in a Nine-grid for a piece type.

        Args:
            i, j: Nine-grid position
            row, col: Optional cell to simulate adding (0 means don't add)
            piece: "X" or "O"

        Returns:
            (total_points, row_points, col_points, diag_points)
        """
        nine_grid = self.coord_to_str(i, j)
        all_pieces = self.piece_coord[nine_grid][piece].copy()

        if row > 0 and col > 0:
            all_pieces.append(self.coord_to_str(row, col))

        point = 0
        point_row = 0
        point_col = 0
        point_diag = 0

        if len(all_pieces) >= 3:
            # Check rows
            for s in range(1, 4):
                if all(self.coord_to_str(s, t) in all_pieces for t in range(1, 4)):
                    point += 1
                    point_row += 1

            # Check columns
            for t in range(1, 4):
                if all(self.coord_to_str(s, t) in all_pieces for s in range(1, 4)):
                    point += 1
                    point_col += 1

            # Check diagonals
            if all(self.coord_to_str(s, s) in all_pieces for s in range(1, 4)):
                point += 1
                point_diag += 1
            if all(self.coord_to_str(s, 4 - s) in all_pieces for s in range(1, 4)):
                point += 1
                point_diag += 1

        return point, point_row, point_col, point_diag

    def current_piece(self, step: int) -> str:
        """Get current player's piece type based on step number."""
        return "X" if step % 2 == 1 else "O"

    def generate_uttt_game(self, plot_level: str = "Easy") -> dict | None:
        """Generate a valid game state with optimal next move.

        Returns:
            Game state dict if successful, None otherwise
        """
        step = 1
        step_limit = 81
        break_step = self.break_step_dict[plot_level]["low"]
        last_i, last_j = 2, 2  # Start in middle Nine-grid
        middle_cell_count = 0

        while step <= step_limit:
            piece = self.current_piece(step)
            max_point_diff, best_pieces = self.find_best_piece_in_nine_grid(
                last_i, last_j, piece
            )

            # Check if we found a unique optimal move
            if len(best_pieces) == 1 and max_point_diff > 0 and step >= break_step:
                row, col = best_pieces[0]
                self.best_next_step = [last_i, last_j, row, col]
                break
            else:
                row, col = random.choice(best_pieces)
                self.draw_piece(last_i, last_j, row, col, piece)
                self.last_step = [last_i, last_j, row, col]

                if row == 2 and col == 2:
                    middle_cell_count += 1
                if middle_cell_count >= 9:
                    return None  # Game over
                if row == 0 or col == 0:
                    return None  # Invalid move

                if max_point_diff > 0:
                    nine_grid = self.coord_to_str(last_i, last_j)
                    self.point_record[piece][nine_grid] += max_point_diff
                    self.all_points[piece] += max_point_diff

                step += 1
                last_i, last_j = row, col

        return self.get_game_state()

    def get_game_state(self) -> dict:
        """Get current game state as dictionary."""
        game_state = {
            "rows": 3,
            "cols": 3,
            "middle_cell_count": self.middle_cell_count,
            "last_step": self.last_step,
            "best_next_step": self.best_next_step,
            "total_steps": sum(
                len(self.piece_coord[coord]["X"]) + len(self.piece_coord[coord]["O"])
                for coord in self.piece_coord
            ),
            "piece_info": [],
        }

        for coord, pieces in self.piece_coord.items():
            for piece_type, coords in pieces.items():
                for coord_str in coords:
                    game_state["piece_info"].append(
                        {"nine_grid": coord, "position": coord_str, "type": piece_type}
                    )

        return game_state


def check_point_in_nine_grid_static(
    nine_grid: str, all_pieces: list[str]
) -> tuple[int, int, int, int]:
    """Static helper to calculate points in a Nine-grid.

    Args:
        nine_grid: Nine-grid coordinate string like "(1, 2)"
        all_pieces: List of piece position strings like "(2, 3)"

    Returns:
        (total_points, row_points, col_points, diag_points)
    """
    point = 0
    point_row = 0
    point_col = 0
    point_diag = 0

    if len(all_pieces) >= 3:
        # Check rows
        for s in range(1, 4):
            if all(f"({s}, {t})" in all_pieces for t in range(1, 4)):
                point += 1
                point_row += 1

        # Check columns
        for t in range(1, 4):
            if all(f"({s}, {t})" in all_pieces for s in range(1, 4)):
                point += 1
                point_col += 1

        # Check diagonals
        if all(f"({s}, {s})" in all_pieces for s in range(1, 4)):
            point += 1
            point_diag += 1
        if all(f"({s}, {4 - s})" in all_pieces for s in range(1, 4)):
            point += 1
            point_diag += 1

    return point, point_row, point_col, point_diag


def trans_coord_to_str(nine_grid: str, position: str) -> str:
    """Convert Nine-grid and position strings to full coordinate string.

    Args:
        nine_grid: "(i, j)"
        position: "(row, col)"

    Returns:
        "(i, j, row, col)"
    """
    i, j = map(int, nine_grid[1:-1].split(", "))
    row, col = map(int, position[1:-1].split(", "))
    return f"({i}, {j}, {row}, {col})"


class GameRLUltraTicTacToeQAEnv(Env):
    """Ultra TicTacToe Question-Answering Environment.

    A single-turn QA environment featuring a 3x3 grid of TicTacToe boards.
    Questions test understanding of game state, rules, and optimal strategy.

    Question Types:
    - Type 1 (Easy): Which player marked a given cell? (3 options)
    - Type 2 (Easy): Number of possible next coordinates (10 options)
    - Type 3 (Easy): Number of marked middle cells (10 options)
    - Type 4 (Medium): Total number of pieces (8 options)
    - Type 5 (Medium): Player's points in a Nine-grid (8 options)
    - Type 6 (Hard): Which Nine-grid for next move? (9 options)
    - Type 7 (Hard): Optimal coordinate for highest point (fill-in)

    Args:
        plot_level: Game complexity ("Easy", "Medium", "Hard")
        question_type: Specific question type (1-7), or None for random
    """

    QUESTION_TYPES = [
        {
            "id": "state_info_marked",
            "name": "State Info: Cell Marked By",
            "level": "Easy",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 3,
        },
        {
            "id": "state_info_next_coords",
            "name": "State Info: Next Coord Count",
            "level": "Easy",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 10,
        },
        {
            "id": "state_info_middle_cells",
            "name": "State Info: Middle Cell Count",
            "level": "Easy",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 10,
        },
        {
            "id": "state_info_total_pieces",
            "name": "State Info: Total Pieces",
            "level": "Medium",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 8,
        },
        {
            "id": "state_info_points",
            "name": "State Info: Player Points",
            "level": "Medium",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 8,
        },
        {
            "id": "state_prediction",
            "name": "State Prediction: Next Nine-grid",
            "level": "Hard",
            "answer_format": "single_choice",
            "qa_type": "StatePrediction",
            "options": 9,
        },
        {
            "id": "strategy_optimization",
            "name": "Strategy Optimization: Optimal Coord",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "StrategyOptimization",
            "options": None,
        },
    ]

    def __init__(
        self, plot_level: str = "Easy", question_type: int | None = None, **kwargs
    ):
        """Initialize Ultra TicTacToe QA environment.

        Args:
            plot_level: Game complexity level
            question_type: Specific question type (1-7), or None for random
            **kwargs: Additional arguments passed to base Env class
        """
        super().__init__(**kwargs)
        self._plot_level = plot_level
        self._question_type = question_type
        self._game = None
        self._game_state = None
        self._current_question = None

    @property
    def description(self) -> str:
        """Return environment description with game rules."""
        return "Ultra TicTacToe QA\n\n" + GAME_RULES

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Reset environment and generate a new question.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)

        # Generate valid game state
        self._game = self._generate_valid_game()
        self._game_state = self._game.get_game_state()

        # Generate question
        q_type = (
            self._question_type
            if self._question_type is not None
            else random.randint(1, 7)
        )
        self._current_question = self._generate_question(q_type)

        obs = Observation(image=self.render(), text=self._current_question["question"])

        return obs, {}

    def _generate_valid_game(self) -> UTTTGameGrid:
        """Generate a valid game state meeting difficulty requirements.

        Returns:
            UTTTGameGrid instance with valid game state
        """
        while True:
            grid = UTTTGameGrid(plot_level=self._plot_level)
            game_state = grid.generate_uttt_game(plot_level=self._plot_level)

            if game_state is None:
                continue

            total_steps = game_state["total_steps"]
            step_range = grid.break_step_dict[self._plot_level]

            if step_range["low"] <= total_steps <= step_range["high"]:
                return grid

    def _generate_question(self, q_type: int) -> dict[str, Any]:
        """Generate a question of the specified type.

        Args:
            q_type: Question type (1-7)

        Returns:
            Dictionary with 'question', 'answer', and 'analysis' keys
        """
        question_generators = {
            1: self._generate_question_type_1,
            2: self._generate_question_type_2,
            3: self._generate_question_type_3,
            4: self._generate_question_type_4,
            5: self._generate_question_type_5,
            6: self._generate_question_type_6,
            7: self._generate_question_type_7,
        }

        if q_type not in question_generators:
            raise ValueError(f"Invalid question type: {q_type}")

        return question_generators[q_type]()

    def _generate_question_type_1(self) -> dict[str, Any]:
        """Type 1: Which player marked the cell at a given coordinate?

        Returns:
            Question dict with 3 options (First Player/Second Player/Not Marked)
        """
        # Pick a random Nine-grid and position
        nine_grids = list(
            set(info["nine_grid"] for info in self._game_state["piece_info"])
        )

        if nine_grids:
            nine_grid = random.choice(nine_grids)
            positions = list(
                set(
                    info["position"]
                    for info in self._game_state["piece_info"]
                    if info["nine_grid"] == nine_grid
                )
            )
            if positions:
                position = random.choice(positions)
            else:
                position = random.choice(
                    [f"({row}, {col})" for row in range(1, 4) for col in range(1, 4)]
                )
        else:
            nine_grid = random.choice(
                [f"({i}, {j})" for i in range(1, 4) for j in range(1, 4)]
            )
            position = random.choice(
                [f"({row}, {col})" for row in range(1, 4) for col in range(1, 4)]
            )

        # Determine answer
        block_info = next(
            (
                info
                for info in self._game_state["piece_info"]
                if info["nine_grid"] == nine_grid and info["position"] == position
            ),
            None,
        )

        if block_info:
            block_type = block_info["type"]
            if block_type == "X":
                option_number = "1"
                process = f"There is an X piece at {trans_coord_to_str(nine_grid, position)} in the image, which means it has been marked by First Player."
            else:  # "O"
                option_number = "2"
                process = f"There is an O piece at {trans_coord_to_str(nine_grid, position)} in the image, which means it has been marked by Second Player."
        else:
            option_number = "3"
            process = f"There is no piece at {trans_coord_to_str(nine_grid, position)} in the image, which means it has not been marked by any player."

        coord_str = trans_coord_to_str(nine_grid, position)
        options = ["First Player", "Second Player", "Not Marked"]
        option_list = " ".join([f"{i+1}. {option}" for i, option in enumerate(options)])

        question = f"{GAME_RULES} Which player marked the cell at {coord_str} in the image? Options: {option_list}"
        analysis = f"{process} So, the option number is {option_number}."

        return {"question": question, "answer": option_number, "analysis": analysis}

    def _generate_question_type_2(self) -> dict[str, Any]:
        """Type 2: Given opponent's last move, how many possible next coordinates?

        Returns:
            Question dict with 10 options (0-9)
        """
        last_step = self._game_state["last_step"]
        last_i, last_j, last_row, last_col = last_step
        last_piece_coord = f"({last_i}, {last_j}, {last_row}, {last_col})"

        best_next_step = self._game_state["best_next_step"]
        next_i, next_j, _, _ = best_next_step

        # Count available coordinates in next Nine-grid
        coords_marked_by_X = [
            info["position"]
            for info in self._game_state["piece_info"]
            if info["nine_grid"] == f"({next_i}, {next_j})" and info["type"] == "X"
        ]
        str_coords_marked_by_X = (
            ", ".join(coords_marked_by_X) if coords_marked_by_X else "no cell"
        )

        coords_marked_by_O = [
            info["position"]
            for info in self._game_state["piece_info"]
            if info["nine_grid"] == f"({next_i}, {next_j})" and info["type"] == "O"
        ]
        str_coords_marked_by_O = (
            ", ".join(coords_marked_by_O) if coords_marked_by_O else "no cell"
        )

        coord_marked_list = [
            info["position"]
            for info in self._game_state["piece_info"]
            if info["nine_grid"] == f"({next_i}, {next_j})"
        ]
        coord_avail_list = [
            (x, y)
            for x in range(1, 4)
            for y in range(1, 4)
            if f"({x}, {y})" not in coord_marked_list
        ]
        coord_avail_num = len(coord_avail_list)
        coord_avail_list_str = ", ".join([f"({x}, {y})" for x, y in coord_avail_list])

        # Generate options
        options_made = [str(num) for num in range(0, 10)]
        option_list = " ".join(
            [f"{i+1}. {option}" for i, option in enumerate(options_made)]
        )
        option_number = str(options_made.index(str(coord_avail_num)) + 1)

        question = f"{GAME_RULES} Now your opponent place a piece at {last_piece_coord}. What is the number of possible coordinates of your next step? Options: {option_list}"
        analysis = f"Since the opponent placed a piece at {last_piece_coord}, our next step should be in the Nine-grid ({next_i}, {next_j}). In this nine grid, we can see that {str_coords_marked_by_X} are marked by the First Player, while {str_coords_marked_by_O} are marked by the Second Player. So, the possible coordinates are the rest cells in the Nine-grid, being {coord_avail_list_str}. This means there are {coord_avail_num} available coordinate(s), so the option number is {option_number}."

        return {"question": question, "answer": option_number, "analysis": analysis}

    def _generate_question_type_3(self) -> dict[str, Any]:
        """Type 3: How many middle cells are marked?

        Returns:
            Question dict with 10 options (0-9)
        """
        middle_cells = [
            (f"({i}, {j})", "(2, 2)") for i in range(1, 4) for j in range(1, 4)
        ]

        marked_middle_cell_num = 0
        counting_process = []

        for nine_grid, position in middle_cells:
            piece_info = next(
                (
                    info
                    for info in self._game_state["piece_info"]
                    if info["nine_grid"] == nine_grid and info["position"] == position
                ),
                None,
            )

            if piece_info:
                marked_middle_cell_num += 1
                piece_type = piece_info["type"]
                counting_process.append(
                    f"The middle cell at ({nine_grid[1:-1]}, {position[1:-1]}) is marked by {piece_type}."
                )
            else:
                counting_process.append(
                    f"The middle cell at ({nine_grid[1:-1]}, {position[1:-1]}) is not marked."
                )

        # Generate options
        options_made = [str(num) for num in range(0, 10)]
        option_number = str(options_made.index(str(marked_middle_cell_num)) + 1)
        option_list = " ".join(
            [f"{i+1}. {option}" for i, option in enumerate(options_made)]
        )

        question = f"{GAME_RULES} How many middle cells in the image are marked? Options: {option_list}"
        analysis = f"We check the middle cells in the Nine-grids one by one.\n{chr(10).join(counting_process)}\nSo there are {marked_middle_cell_num} middle cell(s) marked, the option number is {option_number}."

        return {"question": question, "answer": option_number, "analysis": analysis}

    def _generate_question_type_4(self) -> dict[str, Any]:
        """Type 4: How many pieces are there in total?

        Returns:
            Question dict with 8 options
        """
        counting_process = []
        piece_counts = {"X": [], "O": []}
        total_piece_count = 0

        for i in range(1, 4):
            for j in range(1, 4):
                nine_grid = f"({i}, {j})"
                x_count = len(
                    [
                        info
                        for info in self._game_state["piece_info"]
                        if info["nine_grid"] == nine_grid and info["type"] == "X"
                    ]
                )
                o_count = len(
                    [
                        info
                        for info in self._game_state["piece_info"]
                        if info["nine_grid"] == nine_grid and info["type"] == "O"
                    ]
                )
                piece_counts["X"].append(x_count)
                piece_counts["O"].append(o_count)
                total_pieces_in_nine_grid = x_count + o_count
                total_piece_count += total_pieces_in_nine_grid

                if total_pieces_in_nine_grid > 0:
                    counting_process.append(
                        f"In Nine-grid ({i}, {j}), there are {x_count} X piece(s) and {o_count} O piece(s), totaling {total_pieces_in_nine_grid} piece(s)."
                    )
                else:
                    counting_process.append(
                        f"In nine grid ({i}, {j}), there are no pieces."
                    )

        adding_process = " + ".join(
            [
                str(x_count + o_count)
                for x_count, o_count in zip(
                    piece_counts["X"], piece_counts["O"], strict=False
                )
            ]
        )
        if not adding_process.strip().replace(" + ", ""):
            adding_process = "0"

        # Generate 8 options
        options_made = [str(total_piece_count)]
        random_range = 15
        for _ in range(7):
            offset = random.randint(-random_range, random_range)
            while (
                str(total_piece_count + offset) in options_made
                or (total_piece_count + offset) < 0
            ):
                offset = random.randint(-random_range, random_range)
            options_made.append(str(total_piece_count + offset))
        random.shuffle(options_made)

        option_number = str(options_made.index(str(total_piece_count)) + 1)
        option_list = " ".join(
            [f"{i+1}. {option}" for i, option in enumerate(options_made)]
        )

        question = f"{GAME_RULES} How many pieces are there in the image? Options: {option_list}"
        analysis = f"We count the number of chess pieces in the Nine-grids one by one. {chr(10).join(counting_process)} So there are {adding_process} = {total_piece_count} pieces, the option number is {option_number}."

        return {"question": question, "answer": option_number, "analysis": analysis}

    def _generate_question_type_5(self) -> dict[str, Any]:
        """Type 5: How many points has a player got in a Nine-grid?

        Returns:
            Question dict with 8 options
        """
        # Pick random Nine-grid and player
        nine_grids = list(
            set(info["nine_grid"] for info in self._game_state["piece_info"])
        )

        if not nine_grids:
            nine_grid = random.choice(
                [f"({i}, {j})" for i in range(1, 4) for j in range(1, 4)]
            )
        else:
            nine_grid = random.choice(nine_grids)

        player_name = random.choice(["First Player", "Second Player"])
        piece_type = "X" if player_name == "First Player" else "O"

        # Get pieces in Nine-grid
        all_pieces = [
            info["position"]
            for info in self._game_state["piece_info"]
            if info["nine_grid"] == nine_grid and info["type"] == piece_type
        ]

        # Calculate points
        point, point_row, point_col, point_diag = check_point_in_nine_grid_static(
            nine_grid, all_pieces
        )

        # Generate 8 options
        options_made = [str(point)]
        random_range = 15
        for _ in range(7):
            offset = random.randint(-random_range, random_range)
            while str(point + offset) in options_made or (point + offset) < 0:
                offset = random.randint(-random_range, random_range)
            options_made.append(str(point + offset))
        random.shuffle(options_made)

        option_number = str(options_made.index(str(point)) + 1)
        option_list = " ".join(
            [f"{i+1}. {option}" for i, option in enumerate(options_made)]
        )

        question = f"{GAME_RULES} How many points has the {player_name} got within the Nine-grid {nine_grid}? Options: {option_list}"
        analysis = f"The {player_name} uses {piece_type} pieces. We count the points in the order of rows, columns, and diagonals. We can see that in Nine-grid {nine_grid}, there are {point_row} point(s) in rows, {point_col} point(s) in columns, and {point_diag} point(s) in diagonals, which is {point} point(s) in total. So, the option number is {option_number}."

        return {"question": question, "answer": option_number, "analysis": analysis}

    def _generate_question_type_6(self) -> dict[str, Any]:
        """Type 6: In which Nine-grid should you place the next piece?

        Returns:
            Question dict with 9 options (all Nine-grids)
        """
        # Determine next Nine-grid from game state
        best_next_step = self._game_state["best_next_step"]
        last_step = self._game_state["last_step"]

        # Determine current player
        other_piece_type = next(
            info["type"]
            for info in self._game_state["piece_info"]
            if info["nine_grid"] == f"({last_step[0]}, {last_step[1]})"
            and info["position"] == f"({last_step[2]}, {last_step[3]})"
        )
        piece_type = "X" if other_piece_type == "O" else "O"
        player_name = "First Player" if piece_type == "X" else "Second Player"
        supp_for_X = (
            "plus the first chess piece is in the Nine-grid (2, 2) and there is no corresponding previous step O piece, "
            if player_name == "First Player"
            else ""
        )

        # Count pieces for analysis
        your_piece_counts = {}
        other_piece_counts = {
            f"({row}, {col})": [0] * 9 for row in range(1, 4) for col in range(1, 4)
        }

        for nine_grid in [f"({i}, {j})" for i in range(1, 4) for j in range(1, 4)]:
            your_pieces = [
                info["position"]
                for info in self._game_state["piece_info"]
                if info["nine_grid"] == nine_grid and info["type"] == piece_type
            ]
            your_piece_counts[nine_grid] = len(your_pieces)

            for pos in [
                f"({row}, {col})" for row in range(1, 4) for col in range(1, 4)
            ]:
                other_pieces = [
                    info["position"]
                    for info in self._game_state["piece_info"]
                    if info["nine_grid"] == nine_grid
                    and info["type"] == other_piece_type
                    and info["position"] == pos
                ]
                grid_idx = (int(nine_grid[1]) - 1) * 3 + int(nine_grid[4]) - 1
                other_piece_counts[pos][grid_idx] = len(other_pieces)

        # Generate counting processes
        counting_process_of_your_piece = []
        your_piece_num = []
        for nine_grid in [f"({i}, {j})" for i in range(1, 4) for j in range(1, 4)]:
            your_count = your_piece_counts[nine_grid]
            counting_process_of_your_piece.append(
                f"In nine grid ({nine_grid[1:-1]}), there are {your_count} {piece_type} piece(s)."
            )
            your_piece_num.append(your_count)

        counting_process_of_the_other_piece = []
        opposite_piece_num = []
        for pos in [f"({row}, {col})" for row in range(1, 4) for col in range(1, 4)]:
            total_count = sum(other_piece_counts[pos])
            counting_process_of_the_other_piece.append(
                f"In position ({pos[1:-1]}), there are {total_count} {other_piece_type} piece(s) across all Nine-grids."
            )
            opposite_piece_num.append(total_count)

        diff_piece_num = [
            your - opposite
            for your, opposite in zip(your_piece_num, opposite_piece_num, strict=False)
        ]

        # Generate options
        answer = f"Nine-grid ({best_next_step[0]}, {best_next_step[1]})"
        options_made = [
            f"Nine-grid ({x}, {y})" for x in range(1, 4) for y in range(1, 4)
        ]
        option_number = str(options_made.index(answer) + 1)
        option_list = " ".join(
            [f"{i+1}. {option}" for i, option in enumerate(options_made)]
        )

        question = f"{GAME_RULES} If you are {player_name}, from the image, we can see now it's your turn to place a piece. According to the rules of the game, in which Nine-grid should you place the next piece? Options: {option_list}"
        analysis = f"Since we are the {player_name} now, we use the {piece_type} piece. First, we need to count the number of {piece_type} pieces in each Nine-grid.\n{chr(10).join(counting_process_of_your_piece)}\nThen, we need to count the number of {other_piece_type} pieces in each position of every Nine-grid.\n{chr(10).join(counting_process_of_the_other_piece)}\nSo the quantitative differences corresponding to these coordinates are {', '.join([str(diff) for diff in diff_piece_num])} respectively.\nFrom this difference, {supp_for_X}we can tell that our next step should be in {answer}, which means the option number is {option_number}."

        return {"question": question, "answer": option_number, "analysis": analysis}

    def _generate_question_type_7(self) -> dict[str, Any]:
        """Type 7: At which coordinate to place next piece for highest point?

        Returns:
            Question dict with fill-in answer format
        """
        last_step = self._game_state["last_step"]
        last_i, last_j, last_row, last_col = last_step
        last_piece_coord = f"({last_i}, {last_j}, {last_row}, {last_col})"

        next_i, next_j = last_row, last_col

        # Find available coordinates
        coord_marked_list = [
            info["position"]
            for info in self._game_state["piece_info"]
            if info["nine_grid"] == f"({next_i}, {next_j})"
        ]
        coord_avail_list = [
            (x, y)
            for x in range(1, 4)
            for y in range(1, 4)
            if f"({x}, {y})" not in coord_marked_list
        ]
        avail_coord_num = len(coord_avail_list)

        # Determine current player
        other_piece_type = next(
            info["type"]
            for info in self._game_state["piece_info"]
            if info["nine_grid"] == f"({last_i}, {last_j})"
            and info["position"] == f"({last_row}, {last_col})"
        )
        piece_type = "X" if other_piece_type == "O" else "O"

        # Calculate points for each coordinate
        counting_process = []
        best_next_step = self._game_state["best_next_step"]
        best_i, best_j, best_row, best_col = best_next_step

        for row, col in coord_avail_list:
            new_pieces = self._game_state["piece_info"] + [
                {
                    "nine_grid": f"({next_i}, {next_j})",
                    "position": f"({row}, {col})",
                    "type": piece_type,
                }
            ]
            all_pieces_for_check = [
                info["position"]
                for info in new_pieces
                if info["nine_grid"] == f"({next_i}, {next_j})"
                and info["type"] == piece_type
            ]
            point, point_row, point_col, point_diag = check_point_in_nine_grid_static(
                f"({next_i}, {next_j})", all_pieces_for_check
            )

            counting_process.append(
                f"If placing at ({next_i}, {next_j}, {row}, {col}), you will get {point_row} point(s) in rows, {point_col} point(s) in columns, and {point_diag} point(s) in diagonals, totaling {point} point(s)."
            )

            if row == best_row and col == best_col:
                max_coord = f"({next_i}, {next_j}, {row}, {col})"
                max_point = point

        question = f"{GAME_RULES} Now your opponent place a piece at {last_piece_coord}. At which coordinate should you place your next piece to win the highest point?"
        analysis = f"Since the opponent placed a piece at {last_piece_coord}, our next step should be in the Nine-grid ({next_i}, {next_j}). In this Nine-grid, {avail_coord_num} coordinate(s) are available, and we count their points one by one. {chr(10).join(counting_process)} We can see that when choosing {max_coord}, the final point is the highest, being {max_point}. So, the answer is {max_coord}."

        return {"question": question, "answer": max_coord, "analysis": analysis}

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Process the answer to the current question.

        Args:
            action: The answer (option number or coordinate string)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self._current_question is None:
            raise RuntimeError("No question has been generated. Call reset() first.")

        # Check answer
        correct = action.strip() == self._current_question["answer"].strip()
        reward = 1.0 if correct else 0.0

        # Generate response
        if correct:
            response = "Correct!"
        else:
            response = (
                f"Incorrect. The correct answer is: {self._current_question['answer']}\n\n"
                f"{self._current_question['analysis']}"
            )

        obs = Observation(image=self.render(), text=response)

        return obs, reward, True, False, {}

    def render(self) -> Image.Image:
        """Render the Ultra TicTacToe board as a PIL Image.

        Returns:
            PIL Image showing the 3x3 grid of 3x3 boards
        """
        margin = 30
        cell_size = 40
        gap = cell_size  # Gap between Nine-grids
        screen_size = 3 * cell_size * 3 + 2 * gap
        total_size = screen_size + 2 * margin

        # Create white background
        img = Image.new("RGB", (total_size, total_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try to load font, fall back to default if not available
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24
            )
        except OSError:
            font = ImageFont.load_default()

        # Draw main grid (Nine-grid borders)
        for x in range(3):
            for y in range(3):
                top_left_x = margin + x * (3 * cell_size + gap)
                top_left_y = margin + y * (3 * cell_size + gap)
                draw.rectangle(
                    [
                        top_left_x,
                        top_left_y,
                        top_left_x + 3 * cell_size,
                        top_left_y + 3 * cell_size,
                    ],
                    outline=(0, 0, 0),
                    width=2,
                )

        # Draw small grid (cells within Nine-grids)
        for x in range(3):
            for y in range(3):
                top_left_x = margin + x * (3 * cell_size + gap)
                top_left_y = margin + y * (3 * cell_size + gap)
                for k in range(1, 3):
                    # Vertical lines
                    draw.line(
                        [
                            (top_left_x + k * cell_size, top_left_y),
                            (top_left_x + k * cell_size, top_left_y + 3 * cell_size),
                        ],
                        fill=(0, 0, 0),
                        width=1,
                    )
                    # Horizontal lines
                    draw.line(
                        [
                            (top_left_x, top_left_y + k * cell_size),
                            (top_left_x + 3 * cell_size, top_left_y + k * cell_size),
                        ],
                        fill=(0, 0, 0),
                        width=1,
                    )

        # Draw labels
        for s in range(3):
            label = str(s + 1)
            for t in range(3):
                # Row labels
                label_x = margin // 2
                label_y = (
                    margin
                    + t * (3 * cell_size + gap)
                    + s * cell_size
                    + cell_size // 2
                    - 8
                )
                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

                # Column labels
                label_x = (
                    margin
                    + t * (3 * cell_size + gap)
                    + s * cell_size
                    + cell_size // 2
                    - 8
                )
                label_y = margin // 2
                draw.text((label_x, label_y), label, fill=(0, 0, 0), font=font)

        # Draw pieces
        for piece_info in self._game_state["piece_info"]:
            nine_grid = piece_info["nine_grid"]
            position = piece_info["position"]
            piece_type = piece_info["type"]

            i, j = map(int, nine_grid[1:-1].split(", "))
            row, col = map(int, position[1:-1].split(", "))

            # Calculate pixel position
            x = (j - 1) * 3 + col - 1
            y = (i - 1) * 3 + row - 1
            center_x = margin + x * cell_size + cell_size // 2 + (j - 1) * gap
            center_y = margin + y * cell_size + cell_size // 2 + (i - 1) * gap

            if piece_type == "X":
                # Red X
                draw.line(
                    [(center_x - 20, center_y - 20), (center_x + 20, center_y + 20)],
                    fill=(255, 0, 0),
                    width=2,
                )
                draw.line(
                    [(center_x + 20, center_y - 20), (center_x - 20, center_y + 20)],
                    fill=(255, 0, 0),
                    width=2,
                )
            else:  # "O"
                # Blue O
                draw.ellipse(
                    [center_x - 20, center_y - 20, center_x + 20, center_y + 20],
                    outline=(0, 0, 255),
                    width=2,
                )

        return img

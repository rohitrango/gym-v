"""VGRP-Bench puzzle factories - all in one file."""

import argparse
import json
import os
import random
from typing import Any


class Constraint:
    def __init__(self) -> None:
        self.name = ""

    def check(self, game_state: dict[str, Any]) -> bool:
        pass


class ConstraintRowNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        for row in board:
            row_tmp = [cell for cell in row if cell != 0]
            if len(set(row_tmp)) != len(row_tmp):
                return False
        return True


class ConstraintColNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        for col in range(len(board[0])):
            col_tmp = [
                board[row][col] for row in range(len(board)) if board[row][col] != 0
            ]
            if len(set(col_tmp)) != len(col_tmp):
                return False
        return True


class ConstraintSubGridNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_sub_grid_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        assert len(board) == len(board[0]), "board is not square"
        assert len(board) in [4, 9], "board size is not 4 or 9"

        sub_grid_size = int(len(board) ** 0.5)
        for i in range(0, len(board), sub_grid_size):
            for j in range(0, len(board[0]), sub_grid_size):
                sub_grid = [
                    board[x][y]
                    for x in range(i, i + sub_grid_size)
                    for y in range(j, j + sub_grid_size)
                    if board[x][y] != 0
                ]
                if len(set(sub_grid)) != len(sub_grid):
                    return False
        return True


# =============================================================================
# Puzzle Factory Base Class
# =============================================================================


def hint_type(value):
    if value == "random":
        return "random"
    try:
        return int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be 'random' or an integer"
        ) from err


class PuzzleFactory:
    def __init__(self) -> None:
        self.constraints = []
        self.game_name = "unknown"
        self.size = 0
        # Define dataset split ratios (must sum to 10)
        self.train_ratio = 8
        self.val_ratio = 1
        self.ablation_ratio = 1

    def sample_hints(
        self, board: list[list[int]], num_sample_hints: int
    ) -> list[list[int]]:
        # Create a new board filled with zeros
        new_board = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]
        # Sample num_sample_hints cells to keep from the original board
        sampled_cells = random.sample(
            range(len(board) * len(board[0])), num_sample_hints
        )
        for cell in sampled_cells:
            row = cell // len(board[0])
            col = cell % len(board[0])
            new_board[row][col] = board[row][
                col
            ]  # Copy only the sampled cells from original board
        return new_board

    def save_puzzles(
        self,
        puzzles: list[dict[str, Any]],
        save_path: str = "datasets/",
        filename: str = None,
    ) -> None:
        """
        Save the generated puzzles to JSON files, split into train, val, and ablation sets.
        Splits are based on unique solutions with ratios defined in __init__.
        Val set has different solutions from train, while ablation shares solutions with train.
        """
        if filename is None:
            base_path = f"{save_path}/{self.game_name}_{self.size}x{self.size}_puzzles"
        else:
            base_path = f"{save_path}/{filename.rsplit('.', 1)[0]}"

        # Group puzzles by their solutions
        solution_groups = {}
        for puzzle in puzzles:
            solution_key = str(puzzle["solution"])  # Convert to string for dict key
            if solution_key not in solution_groups:
                solution_groups[solution_key] = []
            solution_groups[solution_key].append(puzzle)

        # Sort groups (common groups first to validation set) by size for better distribution
        sorted_groups = sorted(
            solution_groups.items(), key=lambda x: len(x[1]), reverse=True
        )

        # Calculate target sizes based on ratios
        total_puzzles = len(puzzles)
        target_val_size = total_puzzles * self.val_ratio // 10
        target_ablation_size = total_puzzles * self.ablation_ratio // 10

        # Initialize sets
        train_puzzles = []
        val_puzzles = []
        ablation_puzzles = []

        # First, fill validation set with complete groups
        val_solutions = set()
        current_val_size = 0
        val_group_idx = 0

        while val_group_idx < len(sorted_groups) and current_val_size < target_val_size:
            group = sorted_groups[val_group_idx][1]
            if (
                current_val_size + len(group) <= target_val_size * 1.2
            ):  # Allow 20% overflow
                val_puzzles.extend(group)
                val_solutions.add(sorted_groups[val_group_idx][0])
                current_val_size += len(group)
            val_group_idx += 1

        # Fill train and ablation sets with remaining groups
        train_solutions = set()
        current_ablation_size = 0

        for solution, group in sorted_groups:
            if solution in val_solutions:
                continue

            train_solutions.add(solution)
            # Randomly split each remaining group between train and ablation
            if current_ablation_size < target_ablation_size:
                # Calculate how many puzzles we can still add to ablation
                space_left = target_ablation_size - current_ablation_size
                # Take up to 20% of the current group for ablation
                ablation_count = min(max(1, len(group) // 5), space_left)

                # Randomly select puzzles for ablation
                ablation_indices = random.sample(range(len(group)), ablation_count)
                for i in range(len(group)):
                    if i in ablation_indices:
                        ablation_puzzles.append(group[i])
                        current_ablation_size += 1
                    else:
                        train_puzzles.append(group[i])
            else:
                train_puzzles.extend(group)

        # Shuffle each set before saving
        random.shuffle(train_puzzles)
        random.shuffle(val_puzzles)
        random.shuffle(ablation_puzzles)

        # Create all parent directories
        os.makedirs(os.path.dirname(f"{base_path}_train.json"), exist_ok=True)

        # Save splits to separate files
        for split_name, split_puzzles in [
            ("train", train_puzzles),
            ("val", val_puzzles),
            ("ablation", ablation_puzzles),
        ]:
            split_path = f"{base_path}_{split_name}.json"
            with open(split_path, "w") as f:
                json.dump(split_puzzles, f, indent=2)

        print(f"\nSplit and saved {len(puzzles)} puzzles:")
        print(
            f"Train: {len(train_puzzles)} puzzles ({len(train_solutions)} unique solutions)"
        )
        print(
            f"Val: {len(val_puzzles)} puzzles ({len(val_solutions)} unique solutions)"
        )
        print(
            f"Ablation: {len(ablation_puzzles)} puzzles (solutions shared with train)"
        )
        print(f"Files saved to {base_path}_[train/val/ablation].json")

    def check(self, game_state: dict[str, Any]) -> bool:
        for constraint in self.constraints:
            if not constraint.check(game_state):
                return False
        return True

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        pass


# =============================================================================
# Binairo Puzzle Factory
# =============================================================================


class ConstraintRowBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_row_balance"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        assert all(
            all(cell != "*" for cell in row) for row in board
        ), "'*' should be replaced by '0' in the initialization board"

        for row in board:
            if 0 not in row:  # Only check completed rows
                white_count = sum(1 for x in row if x == "w")
                black_count = sum(1 for x in row if x == "b")
                if white_count != black_count or white_count != expected_count:
                    return False
        return True


class ConstraintColBalance(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_col_balance"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        expected_count = size // 2

        for col in range(size):
            column = [board[row][col] for row in range(size)]
            if 0 not in column and "*" not in column:  # Only check completed columns
                white_count = sum(1 for x in column if x == "w")
                black_count = sum(1 for x in column if x == "b")
                if white_count != black_count or white_count != expected_count:
                    return False
        return True


class ConstraintNoTripleAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_no_triple_adjacent"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check rows
        for row in range(size):
            for col in range(size - 2):
                if (
                    board[row][col] != 0
                    and board[row][col] == board[row][col + 1] == board[row][col + 2]
                ):
                    return False

        # Check columns
        for col in range(size):
            for row in range(size - 2):
                if (
                    board[row][col] != 0
                    and board[row][col] == board[row + 1][col] == board[row + 2][col]
                ):
                    return False
        return True


class BinairoPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4 or size % 2 != 0:
            raise ValueError("Size must be an even number greater than or equal to 4")

        self.game_name = "binairo"
        self.size = size
        self.constraints = [
            ConstraintRowBalance(),
            ConstraintColBalance(),
            ConstraintNoTripleAdjacent(),
            # ConstraintUniqueLines()
        ]

        self.all_possible_values = ["w", "b"]  # 'w' for white, 'b' for black

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Futoshiki Puzzle Factory
# =============================================================================


class ConstraintInequality(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_inequality"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)
        inequalities = game_state.get("inequalities", {"row": [], "col": []})

        # Check row inequalities
        row_ineq = inequalities.get(
            "row", [["" for _ in range(size - 1)] for _ in range(size)]
        )
        for row in range(size):
            for col in range(size - 1):
                if row_ineq[row][col] == "<":
                    if board[row][col] != 0 and board[row][col + 1] != 0:
                        if board[row][col] >= board[row][col + 1]:
                            return False
                elif row_ineq[row][col] == ">":
                    if board[row][col] != 0 and board[row][col + 1] != 0:
                        if board[row][col] <= board[row][col + 1]:
                            return False

        # Check column inequalities
        col_ineq = inequalities.get(
            "col", [["" for _ in range(size)] for _ in range(size - 1)]
        )
        for row in range(size - 1):
            for col in range(size):
                if col_ineq[row][col] == "^":
                    if board[row][col] != 0 and board[row + 1][col] != 0:
                        if board[row][col] >= board[row + 1][col]:
                            return False
                elif col_ineq[row][col] == "v":
                    if board[row][col] != 0 and board[row + 1][col] != 0:
                        if board[row][col] <= board[row + 1][col]:
                            return False

        return True


class FutoshikiPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 3 or size > 12:
            raise ValueError("Grid size must be between 3 and 9")

        self.game_name = "futoshiki"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintInequality(),
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Hitori Puzzle Factory
# =============================================================================


class ConstraintHitoriNoRepeat(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_no_repeat"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]  # This is the shading state
        numbers = game_state.get("numbers", [])  # Get the numbers from additional state
        size = len(board)

        # Check rows and columns for unshaded duplicates
        for i in range(size):
            row_values = [
                numbers[i][j] for j in range(size) if board[i][j] == "e"
            ]  # 'e' means unshaded
            col_values = [numbers[j][i] for j in range(size) if board[j][i] == "e"]

            if len(row_values) != len(set(row_values)) or len(col_values) != len(
                set(col_values)
            ):
                return False
        return True


class ConstraintHitoriAdjacent(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_adjacent"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for row in range(size):
            for col in range(size):
                if board[row][col] == "s":  # shaded cell
                    # Check adjacent cells
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == "s":
                            return False
        return True


class ConstraintHitoriConnected(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_hitori_connected"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Find first unshaded or undecided cell
        start = None
        for r in range(size):
            for c in range(size):
                if board[r][c] in ["e", 0]:  # 'e' means unshaded, 0 means undecided
                    start = (r, c)
                    break
            if start:
                break

        if not start:
            return False

        # BFS to check connectivity
        visited = [[False] * size for _ in range(size)]
        queue = [start]
        visited[start[0]][start[1]] = True

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < size
                    and 0 <= nc < size
                    and not visited[nr][nc]
                    and board[nr][nc] in ["e", 0]
                ):
                    visited[nr][nc] = True
                    queue.append((nr, nc))

        # Check if all unshaded and undecided cells are visited
        for r in range(size):
            for c in range(size):
                if board[r][c] in ["e", 0] and not visited[r][c]:
                    return False
        return True


class HitoriPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()

        self.game_name = "hitori"
        self.size = size
        self.constraints = [
            ConstraintHitoriNoRepeat(),
            ConstraintHitoriAdjacent(),
            ConstraintHitoriConnected(),
        ]
        self.all_possible_values = ["e", "s"]  # 'e' for empty/unshaded, 's' for shaded

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Thermometers Puzzle Factory
# =============================================================================


class ConstraintThermometerFill(Constraint):
    """Check if thermometers are filled correctly (from bulb to top, no gaps)"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_fill"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        thermometers = game_state.get("clues", {}).get(
            "thermometers", []
        )  # Fixed: get thermometers from clues

        # Create a set of all thermometer positions for efficient lookup
        thermometer_positions = {(r, c) for therm in thermometers for r, c in therm}

        # Check non-thermometer cells are empty or undefined
        for i in range(len(board)):
            for j in range(len(board[i])):
                if (i, j) not in thermometer_positions and board[i][j] == "s":
                    return False

        # Check thermometer filling rules
        for thermometer in thermometers:
            # Find first empty cell in thermometer
            first_empty = -1
            for i, (r, c) in enumerate(thermometer):
                if board[r][c] == "e":  # if empty
                    first_empty = i
                    break

            # After first empty, all cells must be empty
            if first_empty != -1:
                for i, (r, c) in enumerate(thermometer):
                    if i > first_empty and board[r][c] == "s":  # if selected
                        return False
        return True


class ConstraintThermometerCount(Constraint):
    """Check if row and column counts match the clues"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_thermometer_count"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True

        size = len(board)
        row_counts = clues["row_counts"]
        col_counts = clues["col_counts"]

        # Check rows
        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            if 0 not in board[i]:  # if row is complete
                if row_selected != row_counts[i]:
                    return False
            else:  # if row is incomplete
                if row_selected > row_counts[i]:  # too many selected
                    return False
                if (
                    row_selected + row_undefined < row_counts[i]
                ):  # impossible to reach target
                    return False

        # Check columns
        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(board[i][j] != 0 for i in range(size)):  # if column is complete
                if col_selected != col_counts[j]:
                    return False
            else:  # if column is incomplete
                if col_selected > col_counts[j]:  # too many selected
                    return False
                if (
                    col_selected + col_undefined < col_counts[j]
                ):  # impossible to reach target
                    return False

        return True


class ThermometersPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4:
            raise ValueError("Size must be at least 4")

        self.game_name = "thermometers"
        self.size = size
        self.constraints = [ConstraintThermometerFill(), ConstraintThermometerCount()]

        self.all_possible_values = ["e", "s"]  # empty or selected

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Treesandtents Puzzle Factory
# =============================================================================


class ConstraintRowTents(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        # if board[0][0] == 'e' and board[0][1] == 'e':
        #     import ipdb; ipdb.set_trace()
        clues = game_state.get("clues", None)
        if not clues:
            return True

        for i, row in enumerate(board):
            if 0 not in row:  # If row is complete
                tent_count = row.count("tt")
                if tent_count != clues["row_clues"][i]:
                    return False
            else:  # If row is incomplete
                tent_count = row.count("tt")
                if tent_count > clues["row_clues"][i]:
                    return False
        return True


class ConstraintColTents(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        clues = game_state.get("clues", None)
        if not clues:
            return True

        size = len(board)
        for j in range(size):
            col = [board[i][j] for i in range(size)]
            if 0 not in col:  # If column is complete
                tent_count = col.count("tt")
                if tent_count != clues["col_clues"][j]:
                    return False
            else:  # If column is incomplete
                tent_count = col.count("tt")
                if tent_count > clues["col_clues"][j]:
                    return False
        return True


class ConstraintTentTree(Constraint):
    """
    Check if:
    1. Each tent has exactly one adjacent tree (horizontally or vertically)
    2. Each tree has exactly one adjacent tent (horizontally or vertically) when complete
    3. Each tree should have exactly one tent or potential tent spot (empty cell) adjacent
    """

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Keep track of which trees are paired with which tents
        tree_tent_pairs = {}  # tree position -> tent position

        # First, check each tent has exactly one adjacent tree
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    adjacent_trees = []
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Only orthogonal
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tr":
                                adjacent_trees.append((ni, nj))
                    # Each tent must have exactly one adjacent tree
                    if len(adjacent_trees) != 1:
                        return False

                    tree_pos = adjacent_trees[0]

                    tree_tent_pairs[tree_pos] = (i, j)

        # Then, check each tree
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tr":
                    # Count adjacent tents and empty cells
                    adjacent_tents = 0
                    adjacent_non_allocated = 0
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            if board[ni][nj] == "tt":
                                adjacent_tents += 1
                            elif board[ni][nj] == 0:
                                adjacent_non_allocated += 1

                    if adjacent_tents > 1:
                        return False
                    if adjacent_tents == 1:
                        pass
                    if adjacent_tents == 0:
                        if adjacent_non_allocated == 0:
                            return False

        return True


class ConstraintAdjacentTents(Constraint):
    """
    Check if tents are not adjacent (including diagonally).
    """

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check tents are not adjacent (including diagonally)
        for i in range(size):
            for j in range(size):
                if board[i][j] == "tt":
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < size and 0 <= nj < size:
                                if board[ni][nj] == "tt":
                                    return False
        return True


class ConstraintTentTreeCount(Constraint):
    """
    Check if:
    1. Number of tents + unallocated cells >= number of trees (during solving)
    2. Number of tents == number of trees (for completed board)
    """

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]

        num_trees = sum(row.count("tr") for row in board)
        num_tents = sum(row.count("tt") for row in board)
        num_unallocated = sum(row.count(0) for row in board)

        # If board is complete (no unallocated cells)
        if num_unallocated == 0:
            return num_tents == num_trees

        # During solving, ensure we can still potentially place enough tents
        return (num_tents + num_unallocated) >= num_trees


class TreesAndTentsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "treesandtents"
        self.size = size
        assert size >= 3, "Size must be at least 3"

        self.constraints = [
            ConstraintRowTents(),
            ConstraintColTents(),
            ConstraintTentTree(),
            ConstraintAdjacentTents(),
            ConstraintTentTreeCount(),
        ]

        self.all_possible_values = ["tt", "e"]
        self.num_generator_processes = max(
            os.cpu_count() // 2, 1
        )  # Limit to 4 processes or CPU count

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        """Get possible values for a given cell."""
        board = game_state["board"]
        if board[row][col] != 0:  # If cell is already filled
            return []

        possible = []
        original_value = board[row][col]
        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible.append(value)
        board[row][col] = original_value
        return possible


# =============================================================================
# Starbattle Puzzle Factory
# =============================================================================


DEBUG_CONSTRAINT_ERROR = False


class ConstraintRowStar(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]

        for row_idx, row in enumerate(board):
            if 0 not in row:
                star_count = sum(1 for cell in row if cell == "s")
                if star_count != 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(
                            f"RowStar constraint failed: Row {row_idx} has {star_count} stars (expected 1)"
                        )
                    return False
            else:
                star_count = sum(1 for cell in row if cell == "s")
                if star_count > 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(
                            f"RowStar constraint failed: Incomplete row {row_idx} has {star_count} stars (max 1)"
                        )
                    return False
        return True


class ConstraintColStar(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for col in range(size):
            col_values = [board[row][col] for row in range(size)]

            if 0 not in col_values:
                star_count = sum(1 for val in col_values if val == "s")
                if star_count != 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(
                            f"ColStar constraint failed: Column {col} has {star_count} stars (expected 1)"
                        )
                    return False
            else:
                star_count = sum(1 for val in col_values if val == "s")
                if star_count > 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(
                            f"ColStar constraint failed: Incomplete column {col} has {star_count} stars (max 1)"
                        )
                    return False
        return True


class ConstraintRegionStar(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        regions = game_state["regions"]

        size = len(board)
        region_counts = {}
        for i in range(size):
            for j in range(size):
                if board[i][j] == "s":
                    region = regions[i][j]
                    region_counts[region] = region_counts.get(region, 0) + 1
                    if region_counts[region] > 1:
                        if DEBUG_CONSTRAINT_ERROR:
                            print(
                                f"RegionStar constraint failed: Region {region} has {region_counts[region]} stars (max 1)"
                            )
                        return False

        for region_num in set(cell for row in regions for cell in row):
            region_cells = [
                (i, j)
                for i in range(size)
                for j in range(size)
                if regions[i][j] == region_num
            ]
            if all(board[i][j] != 0 for i, j in region_cells):
                if region_counts.get(region_num, 0) != 1:
                    if DEBUG_CONSTRAINT_ERROR:
                        print(
                            f"RegionStar constraint failed: Completed region {region_num} has {region_counts.get(region_num, 0)} stars (expected 1)"
                        )
                    return False
        return True


class ConstraintAdjacentStar(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for row in range(size):
            for col in range(size):
                if board[row][col] == "s":
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            new_row, new_col = row + dr, col + dc
                            if (
                                0 <= new_row < size
                                and 0 <= new_col < size
                                and board[new_row][new_col] == "s"
                            ):
                                if DEBUG_CONSTRAINT_ERROR:
                                    print(
                                        f"AdjacentStar constraint failed: Stars at ({row},{col}) and ({new_row},{new_col}) are adjacent"
                                    )
                                return False
        return True


class StarBattlePuzzleFactory(PuzzleFactory):
    def __init__(self, size: int, num_stars: int = 1) -> None:
        super().__init__()
        self.game_name = "starbattle"
        self.size = size
        self.num_stars = num_stars
        self.colors = [chr(65 + i) for i in range(size)]
        # During generation, only use row, column, and adjacent constraints
        self.constraints = [
            ConstraintRowStar(),
            ConstraintColStar(),
            ConstraintAdjacentStar(),
            ConstraintRegionStar(),
        ]

        self.all_possible_values = ["s", "e"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        """Get possible values ('e' for empty or 's' for star) for a given cell."""
        board = game_state["board"]

        # If the cell is already filled with 'e' or 's', return empty list
        if board[row][col] in ["s", "e"]:
            return []

        # Try both values and return those that don't immediately violate constraints
        possible = []
        for val in ["s", "e"]:
            board[row][col] = val
            if self.check(game_state):
                possible.append(val)
            board[row][col] = 0  # Reset to initial state

        return possible


# =============================================================================
# Battleships Puzzle Factory
# =============================================================================


class ConstraintBattleships(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # Check if ships touch diagonally or orthogonally
        for i in range(size):
            for j in range(size):
                if isinstance(
                    board[i][j], tuple
                ):  # Check if it's a revealed ship with direction
                    ship_cell, direction = board[i][j]
                    # Add direction-specific checks here
                    if direction in "<>-":  # Horizontal ship
                        # Check cells above and below
                        for di in [-1, 1]:
                            if 0 <= i + di < size and board[i + di][j] == "s":
                                return False
                    elif direction in "^V|":  # Vertical ship
                        # Check cells left and right
                        for dj in [-1, 1]:
                            if 0 <= j + dj < size and board[i][j + dj] == "s":
                                return False
                elif board[i][j] == "s":
                    # Regular ship cell checks
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if (
                                0 <= ni < size
                                and 0 <= nj < size
                                and (
                                    board[ni][nj] == "s"
                                    or (
                                        isinstance(board[ni][nj], tuple)
                                        and board[ni][nj][0] == "s"
                                    )
                                )
                                and (di != 0 and dj != 0)
                            ):  # Diagonal check
                                return False
        return True


class ConstraintBattleshipsHints(Constraint):
    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        hints = game_state["hints"]

        row_hints = hints["row_hints"]
        col_hints = hints["col_hints"]
        ships = hints["ships"]
        size = len(board)

        # Calculate total required ship cells from ships configuration
        total_ship_cells_required = sum(
            int(length) * int(count) for length, count in ships.items()
        )
        total_ship_cells_selected = sum(
            1 for i in range(size) for j in range(size) if board[i][j] == "s"
        )
        total_undefined_cells = sum(
            1 for i in range(size) for j in range(size) if board[i][j] == 0
        )

        # Check if we have enough cells (placed + potential) to fit all ships
        if (
            total_ship_cells_selected + total_undefined_cells
            < total_ship_cells_required
        ):
            return False

        # Check if we haven't exceeded the total required ship cells
        if total_ship_cells_selected > total_ship_cells_required:
            return False

        # Check row hints
        for i in range(size):
            row_selected = sum(1 for j in range(size) if board[i][j] == "s")
            row_undefined = sum(1 for j in range(size) if board[i][j] == 0)
            # Consider both undefined (0) and non-revealed water cells for potential ships
            if all(cell != 0 and cell != -1 for cell in board[i]):  # if row is complete
                if row_selected != row_hints[i]:
                    return False
            else:  # if row is incomplete
                if row_selected > row_hints[i]:  # too many selected
                    return False
                if (
                    row_selected + row_undefined < row_hints[i]
                ):  # impossible to reach target
                    return False

        # Check column hints
        for j in range(size):
            col_selected = sum(1 for i in range(size) if board[i][j] == "s")
            col_undefined = sum(1 for i in range(size) if board[i][j] == 0)
            if all(
                board[i][j] != 0 and board[i][j] != -1 for i in range(size)
            ):  # if column is complete
                if col_selected != col_hints[j]:
                    return False
            else:  # if column is incomplete
                if col_selected > col_hints[j]:  # too many selected
                    return False
                if (
                    col_selected + col_undefined < col_hints[j]
                ):  # impossible to reach target
                    return False

        # When all cells are filled, check ship shapes
        if total_undefined_cells == 0:
            # Find all ships by finding connected components
            visited = [[False] * size for _ in range(size)]
            ship_lengths = []

            def get_ship_length(i: int, j: int) -> int:
                if (
                    i < 0
                    or i >= size
                    or j < 0
                    or j >= size
                    or visited[i][j]
                    or board[i][j] != "s"
                ):
                    return 0

                visited[i][j] = True
                length = 1

                # Check if ship is horizontal
                if j + 1 < size and board[i][j + 1] == "s":
                    # Add all horizontal cells
                    for col in range(j + 1, size):
                        if board[i][col] != "s":
                            break
                        visited[i][col] = True
                        length += 1
                # Check if ship is vertical
                elif i + 1 < size and board[i + 1][j] == "s":
                    # Add all vertical cells
                    for row in range(i + 1, size):
                        if board[row][j] != "s":
                            break
                        visited[row][j] = True
                        length += 1

                return length

            # Find all ships
            for i in range(size):
                for j in range(size):
                    if not visited[i][j] and board[i][j] == "s":
                        ship_lengths.append(get_ship_length(i, j))

            # Count ships of each length
            ship_counts = {}
            for length in ship_lengths:
                ship_counts[length] = ship_counts.get(length, 0) + 1

            # Verify against required ships
            for length, count in ships.items():
                if ship_counts.get(int(length), 0) != int(count):
                    return False

        return True


class BattleshipsPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "battleships"
        self.size = size

        self.constraints = [ConstraintBattleships(), ConstraintBattleshipsHints()]

        self.all_possible_values = ["e", "s"]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        board = game_state["board"]
        if board[row][col] != 0:  # If cell is already filled
            return []

        possible_values = []
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# Renzoku Puzzle Factory
# =============================================================================


class ConstraintAdjacency(Constraint):
    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacency"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        # always use hints from the game state
        hints = game_state.get("hints")

        # Ensure hints have proper dimensions
        if len(hints.get("row", [])) < size:
            hints["row"] = [["0" for _ in range(size - 1)] for _ in range(size)]
        if len(hints.get("col", [])) < size - 1:
            hints["col"] = [["0" for _ in range(size)] for _ in range(size - 1)]

        # convert board to int (if not 0)
        # Note: In our implementation, board usually contains ints or 0.
        # But for safety we can copy. But deepcopy is expensive.
        # We can just access elements directly assuming they are int-compatible if != 0

        # Check row adjacency hints
        for row in range(size):
            for col in range(size - 1):
                if hints["row"][row][col] == "1":
                    val1 = board[row][col]
                    val2 = board[row][col + 1]
                    if val1 == 0 or val2 == 0:
                        continue
                    if abs(val1 - val2) != 1:
                        return False

        # Check column adjacency hints
        for row in range(size - 1):
            for col in range(size):
                if hints["col"][row][col] == "1":
                    val1 = board[row][col]
                    val2 = board[row + 1][col]
                    if val1 == 0 or val2 == 0:
                        continue
                    if abs(val1 - val2) != 1:
                        return False

        return True


class RenzokuPuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        if size < 4 or size > 12:
            raise ValueError("Grid size must be between 4 and 12")

        self.game_name = "renzoku"
        self.size = size
        self.constraints = [
            ConstraintRowNoRepeat(),
            ConstraintColNoRepeat(),
            ConstraintAdjacency(),
        ]
        self.all_possible_values = [i for i in range(1, size + 1)]

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[int]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values


# =============================================================================
# FieldExplore Puzzle Factory
# =============================================================================


class ConstraintAdjacentNumbers(Constraint):
    """Ensures revealed numbers match adjacent mine counts"""

    def __init__(self) -> None:
        super().__init__()
        self.name = "constraint_adjacent_numbers"

    def check(self, game_state: dict[str, Any]) -> bool:
        board = game_state["board"]
        size = len(board)

        for i in range(size):
            for j in range(size):
                if (
                    isinstance(board[i][j], int) and board[i][j] != 0
                ):  # If cell is a revealed number
                    # Count adjacent mines and undefined cells
                    i_start = max(0, i - 1)
                    i_end = min(size, i + 2)
                    j_start = max(0, j - 1)
                    j_end = min(size, j + 2)

                    adjacent_mines = sum(
                        1
                        for r in range(i_start, i_end)
                        for c in range(j_start, j_end)
                        if board[r][c] == "s"
                    )

                    adjacent_undefined = sum(
                        1
                        for r in range(i_start, i_end)
                        for c in range(j_start, j_end)
                        if board[r][c] == 0
                    )

                    # Check if current mines <= number <= potential mines (current + undefined)
                    if (
                        adjacent_mines > board[i][j]
                        or adjacent_mines + adjacent_undefined < board[i][j]
                    ):
                        return False
        return True


class FieldExplorePuzzleFactory(PuzzleFactory):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.game_name = "fieldexplore"
        self.size = size
        self.constraints = [ConstraintAdjacentNumbers()]
        self.all_possible_values = ["s", "e"]  # True for 's', False for 'e'

    def get_possible_values(
        self, game_state: dict[str, Any], row: int, col: int
    ) -> list[str]:
        possible_values = []
        board = game_state["board"]
        original_value = board[row][col]

        for value in self.all_possible_values:
            board[row][col] = value
            if self.check(game_state):
                possible_values.append(value)
        board[row][col] = original_value
        return possible_values

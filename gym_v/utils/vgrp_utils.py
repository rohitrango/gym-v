import copy
import random
from typing import Any


def solve_puzzle_backtrack(
    factory, game_state: dict[str, Any], max_attempts: int = 1000
) -> bool:
    """Simple backtracking solver."""
    board = game_state["board"]
    size = len(board)

    if not hasattr(factory, "_solve_attempts"):
        factory._solve_attempts = 0

    factory._solve_attempts += 1
    if factory._solve_attempts > max_attempts:
        return False

    # Find first empty cell (0)
    for row in range(size):
        for col in range(size):
            if board[row][col] == 0:
                # Try each possible value
                possible_values = factory.get_possible_values(game_state, row, col)
                random.shuffle(possible_values)

                for value in possible_values:
                    board[row][col] = value
                    if solve_puzzle_backtrack(factory, game_state, max_attempts):
                        return True
                    board[row][col] = 0
                return False

    return factory.check(game_state)  # All cells filled, verify final state


def generate_puzzle(
    factory,
    size: int,
    num_hints: int,
    max_attempts: int = 100,
    initial_board: list[list[Any]] | None = None,
    **extra_state,
) -> tuple[list[list[Any]], list[list[Any]]] | None:
    """Generate a puzzle with solution using the factory.

    Args:
        factory: The PuzzleFactory to use.
        size: Grid size.
        num_hints: Number of hints to reveal in the puzzle (if applicable).
        max_attempts: Max attempts to generate a valid solution.
        initial_board: Optional initial board state (e.g. containing trees).
        **extra_state: Additional state for the constraints (e.g. regions, clues).
    """
    for _attempt in range(max_attempts):
        # Create board
        if initial_board is not None:
            board = copy.deepcopy(initial_board)
        else:
            board = [[0 for _ in range(size)] for _ in range(size)]

        game_state = {"board": board, **extra_state}

        # Reset solve attempts counter
        factory._solve_attempts = 0

        # Try to fill the board
        if solve_puzzle_backtrack(factory, game_state):
            # Save solution (deep copy to avoid reference issues)
            solution = copy.deepcopy(board)

            # Create puzzle by sampling hints
            if num_hints > 0:
                puzzle = factory.sample_hints(solution, num_hints)
            else:
                puzzle = [[0 for _ in range(size)] for _ in range(size)]

            return puzzle, solution

    return None

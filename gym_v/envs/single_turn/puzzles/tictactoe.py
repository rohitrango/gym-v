"""TicTacToe QA Environment

This module implements a TicTacToe question-answering environment for the gym-v framework.
Players compete on a 3x3 grid, with 'O' (red) always playing first.

The environment supports 3 question types:
1. StateInfo (Easy): Identify the color at a specific position
2. StrategyOptimization (Medium): Determine the optimal move for current player
3. ActionOutcome (Hard): Predict opponent's optimal response after a specific move

Source: /mnt/petrelfs/gujiawei/jiawei/env-v/Game-RL/src/tictactoe/
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw

from gym_v import Env, Observation, get_logger
from gym_v.utils.gamerl_utils import build_description, score_choice

logger = get_logger()


class TicTacToe:
    """TicTacToe game logic with optimal AI strategy.

    Board representation:
    - 0: Empty
    - 1: O (red, first player)
    - -1: X (blue, second player)

    AI uses strict priority ordering:
    1. Win immediately
    2. Block opponent's immediate win
    3. Create double threat (two ways to win)
    4. Block opponent's double threat
    5. Choose first available position
    """

    def __init__(self):
        self.board = [[0] * 3 for _ in range(3)]
        self.current_player = 1  # O always starts

    def reset(self):
        """Reset the board to initial state."""
        self.board = [[0] * 3 for _ in range(3)]
        self.current_player = 1

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid."""
        return 0 <= row < 3 and 0 <= col < 3 and self.board[row][col] == 0

    def make_move(self, row: int, col: int, player: int = None) -> bool:
        """Make a move at (row, col) for the specified player.

        Args:
            row: Row index (0-2)
            col: Column index (0-2)
            player: Player to make the move (1 or -1). If None, uses current_player.

        Returns:
            True if move was valid and made, False otherwise
        """
        if player is None:
            player = self.current_player

        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = player
        self.current_player = -self.current_player
        return True

    def check_winner(self) -> int:
        """Check if there's a winner.

        Returns:
            1 if O wins, -1 if X wins, 0 if no winner
        """
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != 0:
                return row[0]

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != 0:
                return self.board[0][col]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[1][1]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[1][1]

        return 0

    def is_full(self) -> bool:
        """Check if the board is full."""
        return all(self.board[row][col] != 0 for row in range(3) for col in range(3))

    def find_winning_move(self, player: int) -> list[tuple[int, int]]:
        """Find all positions where player can win immediately.

        Args:
            player: Player to check (1 or -1)

        Returns:
            List of (row, col) tuples where player can win
        """
        winning_moves = []
        for row in range(3):
            for col in range(3):
                if self.is_valid_move(row, col):
                    # Try the move
                    self.board[row][col] = player
                    if self.check_winner() == player:
                        winning_moves.append((row, col))
                    # Undo the move
                    self.board[row][col] = 0
        return winning_moves

    def find_double_threat_move(self, player: int) -> tuple[int, int] | None:
        """Find a move that creates a double threat (two ways to win next turn).

        Args:
            player: Player to check (1 or -1)

        Returns:
            (row, col) tuple if such a move exists, None otherwise
        """
        for row in range(3):
            for col in range(3):
                if self.is_valid_move(row, col):
                    # Try the move
                    self.board[row][col] = player
                    winning_moves = self.find_winning_move(player)
                    # Undo the move
                    self.board[row][col] = 0

                    if len(winning_moves) >= 2:
                        return (row, col)
        return None

    def find_best_move(self, player: int) -> tuple[tuple[int, int] | None, str]:
        """Find the best move using priority-based strategy.

        Priority order:
        1. Win immediately
        2. Block opponent's immediate win
        3. Create double threat
        4. Block opponent's double threat
        5. Choose first available position

        Args:
            player: Current player (1 or -1)

        Returns:
            Tuple of ((row, col) or None, explanation string)
        """
        opponent = -player

        # 1. Check for immediate win
        winning_moves = self.find_winning_move(player)
        if winning_moves:
            pos = winning_moves[0]
            return pos, f"Player can win immediately by moving to ({pos[0]}, {pos[1]})"

        # 2. Block opponent's immediate win
        opponent_winning_moves = self.find_winning_move(opponent)
        if len(opponent_winning_moves) > 1:
            return None, "Opponent has multiple winning moves, cannot block all"
        elif len(opponent_winning_moves) == 1:
            pos = opponent_winning_moves[0]
            return pos, f"Must block opponent's winning threat at ({pos[0]}, {pos[1]})"

        # 3. Create double threat
        double_threat_move = self.find_double_threat_move(player)
        if double_threat_move is not None:
            return (
                double_threat_move,
                f"Can create double threat by moving to ({double_threat_move[0]}, {double_threat_move[1]})",
            )

        # 4. Block opponent's double threat
        opponent_double_threat = self.find_double_threat_move(opponent)
        if opponent_double_threat is not None:
            return (
                opponent_double_threat,
                f"Must block opponent's double threat at ({opponent_double_threat[0]}, {opponent_double_threat[1]})",
            )

        # 5. Choose first available position
        for row in range(3):
            for col in range(3):
                if self.is_valid_move(row, col):
                    return (row, col), f"Choose first available position ({row}, {col})"

        return None, "No available moves"

    def copy(self) -> TicTacToe:
        """Create a deep copy of the game state."""
        new_game = TicTacToe()
        new_game.board = [row[:] for row in self.board]
        new_game.current_player = self.current_player
        return new_game


GAME_RULES = dedent("""
    # TicTacToe Game Rules

    ## Basic Setup
    - The game is played on a 3×3 grid
    - Two players: O (red) and X (blue)
    - O always plays first

    ## How to Play
    1. Players take turns placing their mark in an empty cell
    2. The first player to get 3 of their marks in a row (horizontally, vertically, or diagonally) wins
    3. If all 9 cells are filled and no player has won, the game is a draw

    ## Strategy Priority (for optimal play)
    1. **Win immediately** - If you can win on this turn, do it
    2. **Block opponent's win** - If opponent can win next turn, block them
    3. **Create double threat** - Make a move that gives you two ways to win
    4. **Block opponent's double threat** - Prevent opponent from creating a double threat
    5. **Choose any available position** - If none of the above apply, pick the first available spot
""").strip()


class TicTacToeQAEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    # Overrides: interaction_mode=single_turn, action_format=open_ended
    """TicTacToe Question-Answering Environment.

    A single-turn QA environment where the agent must answer questions about
    TicTacToe game states, including state identification, optimal strategy,
    and predicting opponent responses.

    Question Types:
    - Type 0 (Easy): Identify the color at a specific position (3 options)
    - Type 1 (Medium): Determine the optimal move for current player (8 options)
    - Type 2 (Hard): Predict opponent's optimal response after a move (8 options)

    Args:
        question_type: Specific question type to use (0-2), or None for random
    """

    QUESTION_TYPES = [
        {
            "id": "state_info",
            "name": "State Info: Position Color",
            "level": "Easy",
            "answer_format": "single_choice",
            "qa_type": "StateInfo",
            "options": 3,
        },
        {
            "id": "strategy_optimization",
            "name": "Strategy Optimization: Optimal Move",
            "level": "Medium",
            "answer_format": "single_choice",
            "qa_type": "StrategyOptimization",
            "options": 8,
        },
        {
            "id": "action_outcome",
            "name": "Action Outcome: Opponent Response",
            "level": "Hard",
            "answer_format": "single_choice",
            "qa_type": "ActionOutcome",
            "options": 8,
        },
    ]

    def __init__(
        self, question_type: int | None = None, num_players: int = 1, **kwargs
    ):
        """Initialize TicTacToe QA environment.

        Args:
            question_type: Specific question type (0-2), or None for random
            **kwargs: Additional arguments passed to base Env class
        """
        super().__init__(**kwargs)
        self._question_type_param = question_type
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}
        self._game = TicTacToe()

        # Standard QA variables
        self._question_type_idx: int = 0
        self._question: str = ""
        self._options: list[str] | None = None
        self._oracle_answer: str = ""

    @property
    def description(self) -> str:
        """Return game rules + current question + answer format."""
        return build_description(
            game_name="Tic-Tac-Toe",
            rules=GAME_RULES,
            question=self._question,
            options=self._options,
            oracle_answer=self._oracle_answer,
        )

    def _get_state_text(self) -> str:
        """Generate text description of current TicTacToe game state.

        Returns a text representation that contains the same information as the rendered image.
        """
        # Create text grid representation
        grid = []
        for row in range(3):
            row_chars = []
            for col in range(3):
                cell = self._game.board[row][col]
                if cell == 1:
                    row_chars.append("O")
                elif cell == -1:
                    row_chars.append("X")
                else:
                    row_chars.append(".")
            grid.append("".join(row_chars))

        grid_str = "\n".join(grid)

        return f"""Grid Size: 3x3
Grid (O=first player, X=second player, .=empty):
{grid_str}"""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """Reset environment and generate a new question.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)

        # Reset game and create random state
        self._game.reset()
        self._generate_random_state()

        # Generate question
        if self._question_type_param is not None:
            self._question_type_idx = self._question_type_param
        else:
            self._question_type_idx = int(self.np_random.integers(0, 3))
        q_type = self.QUESTION_TYPES[self._question_type_idx]

        # Generate question - sets _question, _options, _oracle_answer
        result = self._generate_question(self._question_type_idx)
        self._question = result["question"]
        self._options = result.get("options")
        self._oracle_answer = result["answer"]

        text_state = self._get_state_text()
        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": text_state,
                "text_prompt": f"{self.description}",
                "question": self._question,
                "options": self._options,
                "question_type": q_type["name"],
                "level": q_type["level"],
            },
        )

        info = {
            "seed": seed,
            "oracle_answer": self._oracle_answer,
            "question_type": q_type["id"],
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def _generate_random_state(self):
        """Generate a random valid game state."""
        # Randomly decide how many moves to make (1-6 moves)
        num_moves = self.py_random.randint(1, 6)

        available_positions = [(r, c) for r in range(3) for c in range(3)]
        self.py_random.shuffle(available_positions)

        for i in range(num_moves):
            if i < len(available_positions):
                row, col = available_positions[i]
                self._game.board[row][col] = 1 if i % 2 == 0 else -1

        # Set current player (alternates based on number of moves)
        self._game.current_player = 1 if num_moves % 2 == 0 else -1

    def _generate_question(self, q_type: int) -> dict[str, Any]:
        """Generate a question of the specified type.

        Args:
            q_type: Question type (0-2)

        Returns:
            Dictionary with 'question', 'answer', and 'analysis' keys
        """
        if q_type == 0:
            return self._generate_question_type_0()
        elif q_type == 1:
            return self._generate_question_type_1()
        elif q_type == 2:
            return self._generate_question_type_2()
        else:
            raise ValueError(f"Invalid question type: {q_type}")

    def _generate_question_type_0(self) -> dict[str, Any]:
        """Type 0: What is the color of the block at (row, col)?

        Returns:
            Question dict with 3 color options (red/blue/white)
        """
        # Pick a random position
        row = self.py_random.randint(0, 2)
        col = self.py_random.randint(0, 2)

        cell_value = self._game.board[row][col]

        # Determine answer
        if cell_value == 1:
            answer = "A"
            color = "red"
        elif cell_value == -1:
            answer = "B"
            color = "blue"
        else:
            answer = "C"
            color = "white"

        question = f"""{GAME_RULES}

**Question:** What is the color of the block at position ({row}, {col})?

A. red
B. blue
C. white"""

        analysis = f"Looking at position ({row}, {col}), we can see it is {color}."

        return {"question": question, "answer": answer, "analysis": analysis}

    def _generate_question_type_1(self) -> dict[str, Any]:
        """Type 1: What is the optimal move for the current player?

        Returns:
            Question dict with 8 position options (A-H) or None
        """
        current_player = self._game.current_player
        player_name = "O (red)" if current_player == 1 else "X (blue)"

        # Find optimal move
        best_move, explanation = self._game.find_best_move(current_player)

        # Generate 8 random positions (including the best move if it exists)
        all_positions = [(r, c) for r in range(3) for c in range(3)]
        self.py_random.shuffle(all_positions)

        # Build options
        options = []
        correct_option = None

        if best_move is None:
            # No valid move exists
            correct_option = "A"
            options.append(("A", "None"))
            for i, pos in enumerate(all_positions[:7]):
                options.append((chr(66 + i), f"({pos[0]}, {pos[1]})"))
        else:
            # Place best move randomly in options
            correct_option = chr(65 + self.py_random.randint(0, 7))

            for i in range(8):
                opt_letter = chr(65 + i)
                if opt_letter == correct_option:
                    options.append((opt_letter, f"({best_move[0]}, {best_move[1]})"))
                else:
                    # Use other positions
                    pos_idx = i if i < all_positions.index(best_move) else i - 1
                    if pos_idx < len(all_positions):
                        pos = all_positions[pos_idx]
                        if pos != best_move:
                            options.append((opt_letter, f"({pos[0]}, {pos[1]})"))

        # Build question text
        question = f"""{GAME_RULES}

**Question:** What is the optimal move for the current player ({player_name})?

"""
        for opt_letter, opt_text in options:
            question += f"{opt_letter}. {opt_text}\n"

        analysis = explanation

        return {
            "question": question.strip(),
            "answer": correct_option,
            "analysis": analysis,
        }

    def _generate_question_type_2(self) -> dict[str, Any]:
        """Type 2: If current player moves to (row, col), what is opponent's optimal response?

        Returns:
            Question dict with 8 position options (A-H) or None
        """
        current_player = self._game.current_player
        opponent = -current_player

        player_name = "O (red)" if current_player == 1 else "X (blue)"
        opponent_name = "X (blue)" if opponent == -1 else "O (red)"

        # Pick a random valid move for current player
        valid_moves = [
            (r, c) for r in range(3) for c in range(3) if self._game.is_valid_move(r, c)
        ]

        if not valid_moves:
            # Game is over, create a fallback question
            move_pos = (0, 0)
            opponent_response = None
            explanation = "The board is full, no moves available"
        else:
            move_pos = self.py_random.choice(valid_moves)

            # Simulate the move
            game_copy = self._game.copy()
            game_copy.make_move(move_pos[0], move_pos[1], current_player)

            # Find opponent's best response
            opponent_response, explanation = game_copy.find_best_move(opponent)

        # Generate 8 random positions (including opponent's response if it exists)
        all_positions = [(r, c) for r in range(3) for c in range(3)]
        self.py_random.shuffle(all_positions)

        # Build options
        options = []
        correct_option = None

        if opponent_response is None:
            # No valid response exists
            correct_option = "A"
            options.append(("A", "None"))
            for i, pos in enumerate(all_positions[:7]):
                options.append((chr(66 + i), f"({pos[0]}, {pos[1]})"))
        else:
            # Place opponent's response randomly in options
            correct_option = chr(65 + self.py_random.randint(0, 7))

            for i in range(8):
                opt_letter = chr(65 + i)
                if opt_letter == correct_option:
                    options.append(
                        (
                            opt_letter,
                            f"({opponent_response[0]}, {opponent_response[1]})",
                        )
                    )
                else:
                    # Use other positions
                    pos_idx = i if i < all_positions.index(opponent_response) else i - 1
                    if pos_idx < len(all_positions):
                        pos = all_positions[pos_idx]
                        if pos != opponent_response:
                            options.append((opt_letter, f"({pos[0]}, {pos[1]})"))

        # Build question text
        question = f"""{GAME_RULES}

**Question:** If the current player ({player_name}) moves to position ({move_pos[0]}, {move_pos[1]}), what is the opponent's ({opponent_name}) optimal response?

"""
        for opt_letter, opt_text in options:
            question += f"{opt_letter}. {opt_text}\n"

        analysis = explanation

        return {
            "question": question.strip(),
            "answer": correct_option,
            "analysis": analysis,
        }

    def _score_answer(self, answer: str) -> float:
        """Score the user's answer.

        Args:
            answer: User's answer string

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        return score_choice(answer, self._oracle_answer)

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Process the answer to the current question.

        Args:
            action: The answer (e.g., "A", "B", "C")

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self._question:
            raise RuntimeError("No question has been generated. Call reset() first.")

        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        # Check answer
        reward = self._score_answer(action_str)
        correct = reward == 1.0

        # Generate response
        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._oracle_answer}"

        obs = Observation(image=self.render(), text=response)

        terminated = True
        truncated = False
        info = {
            "oracle_answer": self._oracle_answer,
            "user_answer": action_str,
            "correct": correct,
        }

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: terminated for agent_id in self._agent_ids},
                "__all__": terminated,
            },
            {
                **{agent_id: truncated for agent_id in self._agent_ids},
                "__all__": truncated,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the TicTacToe board as a PIL Image.

        Returns:
            300x300 PIL Image showing the current board state
        """
        img_size = 300
        cell_size = img_size // 3

        # Create white background
        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw grid lines (black)
        for i in range(1, 3):
            # Vertical lines
            x = i * cell_size
            draw.line([(x, 0), (x, img_size)], fill=(0, 0, 0), width=2)
            # Horizontal lines
            y = i * cell_size
            draw.line([(0, y), (img_size, y)], fill=(0, 0, 0), width=2)

        # Draw pieces
        for row in range(3):
            for col in range(3):
                cell_value = self._game.board[row][col]

                if cell_value != 0:
                    # Calculate cell center
                    x = col * cell_size + cell_size // 2
                    y = row * cell_size + cell_size // 2
                    radius = cell_size // 3

                    if cell_value == 1:
                        # O (red)
                        draw.ellipse(
                            [x - radius, y - radius, x + radius, y + radius],
                            fill=(255, 0, 0),
                            outline=(255, 0, 0),
                        )
                    else:
                        # X (blue) - draw two diagonal lines
                        offset = radius // 2
                        draw.line(
                            [(x - offset, y - offset), (x + offset, y + offset)],
                            fill=(0, 0, 255),
                            width=8,
                        )
                        draw.line(
                            [(x - offset, y + offset), (x + offset, y - offset)],
                            fill=(0, 0, 255),
                            width=8,
                        )

        return img

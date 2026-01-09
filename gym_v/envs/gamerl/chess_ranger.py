"""Chess Ranger QA environment for gym-v.

Chess Ranger is a puzzle where chess pieces are placed on an 8x8 board.
Rules:
- Pieces move like in standard chess
- You can only perform capture moves
- The king is allowed to be captured
- Goal: end up with a single piece remaining on the board

This environment provides 5 question types about the puzzle state and solution.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import random
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation

# ============================================================================
# Core Game Classes (from cr_solver.py)
# ============================================================================


class Square:
    """Represents a square on the chess board."""

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    @staticmethod
    def all() -> Iterator[Square]:
        """Iterate over all 64 squares."""
        for idx in range(64):
            yield Square(idx // 8, idx % 8)

    def is_valid(self) -> bool:
        """Check if square is within board bounds."""
        return 0 <= self.row < 8 and 0 <= self.col < 8

    def row_char(self) -> str:
        """Get row character (8-1)."""
        return chr(ord("8") - self.row)

    def col_char(self) -> str:
        """Get column character (a-h)."""
        return chr(ord("a") + self.col)

    def idx(self) -> int:
        """Get linear index (0-63)."""
        return 8 * self.row + self.col

    def offset(self, row_offset: int, col_offset: int) -> Square | None:
        """Get square at given offset, or None if invalid."""
        new_square = Square(self.row + row_offset, self.col + col_offset)
        return new_square if new_square.is_valid() else None

    def offset_iter(self, row_offset: int, col_offset: int) -> Iterator[Square]:
        """Iterate over squares in a direction until edge."""
        current = self.offset(row_offset, col_offset)
        while current:
            yield current
            current = current.offset(row_offset, col_offset)

    def __str__(self):
        return f"{self.col_char()}{self.row_char()}"


class Piece:
    """Represents a chess piece."""

    PAWN = "Pawn"
    ROOK = "Rook"
    BISHOP = "Bishop"
    KNIGHT = "Knight"
    QUEEN = "Queen"
    KING = "King"

    def __init__(self, name: str):
        self.name = name

    def offsets(self) -> list[tuple[int, int]]:
        """Get movement offsets for this piece type."""
        if self.name == Piece.PAWN:
            return [(-1, -1), (-1, 1)]
        if self.name == Piece.ROOK:
            return [(-1, 0), (0, -1), (0, 1), (1, 0)]
        if self.name == Piece.BISHOP:
            return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        if self.name == Piece.KNIGHT:
            return [
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ]
        if self.name in (Piece.QUEEN, Piece.KING):
            return [
                (-1, 0),
                (0, -1),
                (0, 1),
                (1, 0),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        return []

    def long_range(self) -> bool:
        """Check if piece can move multiple squares in one direction."""
        return self.name in (Piece.ROOK, Piece.BISHOP, Piece.QUEEN)

    def __str__(self):
        return self.name


class Move:
    """Represents a capture move."""

    def __init__(
        self,
        src_piece: Piece,
        src_square: Square,
        dst_piece: Piece | None,
        dst_square: Square,
    ):
        self.src_piece = src_piece
        self.src_square = src_square
        self.dst_piece = dst_piece
        self.dst_square = dst_square

    def __str__(self):
        return f"{self.src_piece} from {self.src_square} to {self.dst_square}"


class Board:
    """Represents the chess board state."""

    def __init__(self, puzzle: str):
        """Initialize board from FEN-like notation.

        Args:
            puzzle: FEN string without spaces, e.g., "8/1K6/8/3B4/..."
        """
        self.board = [None] * 64
        idx = 0
        piece_map = {
            "P": Piece.PAWN,
            "R": Piece.ROOK,
            "B": Piece.BISHOP,
            "N": Piece.KNIGHT,
            "Q": Piece.QUEEN,
            "K": Piece.KING,
        }
        for ch in puzzle:
            if ch.isdigit():
                idx += int(ch)
            elif ch in piece_map:
                self.board[idx] = Piece(piece_map[ch])
                idx += 1

    def piece_count(self) -> int:
        """Count total pieces on board."""
        return sum(1 for square in self.board if square is not None)

    def __getitem__(self, square: Square) -> Piece | None:
        return self.board[square.idx()]

    def __setitem__(self, square: Square, piece: Piece | None):
        self.board[square.idx()] = piece

    def get_piece_positions(self) -> list[tuple[str, str]]:
        """Get list of (piece_name, position_str) for all pieces."""
        pieces = []
        for idx in range(64):
            piece = self.board[idx]
            if piece is not None:
                square = Square(idx // 8, idx % 8)
                pieces.append((piece.name, str(square)))
        return pieces

    def make_move(self, move: Move):
        """Execute a move."""
        self[move.src_square] = None
        self[move.dst_square] = move.src_piece

    def undo_move(self, move: Move):
        """Undo a move."""
        self[move.src_square] = move.src_piece
        self[move.dst_square] = move.dst_piece

    def moves(self) -> list[Move]:
        """Get all legal capture moves."""
        moves = []
        for square in Square.all():
            if self[square] is not None:
                self.collect_moves(square, moves)
        return moves

    def collect_moves(self, src_square: Square, moves: list[Move]):
        """Collect all moves from a source square."""
        src_piece = self[src_square]
        for row_offset, col_offset in src_piece.offsets():
            if src_piece.long_range():
                for dst_square in src_square.offset_iter(row_offset, col_offset):
                    dst_piece = self[dst_square]
                    if dst_piece is not None:
                        moves.append(Move(src_piece, src_square, dst_piece, dst_square))
                        break
            else:
                dst_square = src_square.offset(row_offset, col_offset)
                if dst_square and (dst_piece := self[dst_square]) is not None:
                    moves.append(Move(src_piece, src_square, dst_piece, dst_square))


class Solver:
    """Solves Chess Ranger puzzles using backtracking."""

    def __init__(self, board: Board):
        self.board = board

    def solve(self) -> list[tuple[Piece, Square, Square]] | None:
        """Solve puzzle, return list of moves or None if unsolvable."""
        moves = []
        if self._solve(self.board.piece_count(), moves):
            return [
                (move.src_piece, move.src_square, move.dst_square) for move in moves
            ]
        else:
            return None

    def _solve(self, piece_count: int, moves: list[Move]) -> bool:
        if piece_count == 1:
            return True
        for move in self.board.moves():
            self.board.make_move(move)
            moves.append(move)
            if self._solve(piece_count - 1, moves):
                return True
            moves.pop()
            self.board.undo_move(move)
        return False


class TraceSolver:
    """Solver that records the full backtracking trace."""

    def __init__(self, board: Board):
        self.board = board
        self.trace = []

    def solve(self):
        """Return (moves, trace) where trace contains step-by-step analysis."""
        moves = []
        piece_count = self.board.piece_count()
        self._solve(1, piece_count, moves)
        return moves, self.trace

    def _solve(self, num_step: int, piece_count: int, moves: list[Move]) -> bool:
        if piece_count == 1:
            self.trace.append("Success: Only one piece remains.")
            return True

        for move in self.board.moves():
            self.trace.append(
                f"Try step {num_step}: move {move.src_piece} in {move.src_square} "
                f"to capture {move.dst_piece} in {move.dst_square}."
            )
            self.board.make_move(move)
            moves.append(move)

            if self._solve(num_step + 1, piece_count - 1, moves):
                return True

            moves.pop()
            self.board.undo_move(move)
            self.trace.append(
                f"Backtrack step {num_step}: move {move.src_piece} from {move.dst_square} "
                f"to {move.src_square} and release {move.dst_piece} back to {move.dst_square}."
            )

        self.trace.append(f"Fail: No moves left with {piece_count} pieces.")
        return False


# ============================================================================
# Puzzle Generation
# ============================================================================


def generate_random_puzzle(piece_types: list[str], num_pieces: int) -> str:
    """Generate a random solvable Chess Ranger puzzle.

    Args:
        piece_types: List of piece symbols (e.g., ['P', 'R', 'B', 'N', 'Q', 'K'])
        num_pieces: Number of pieces to place on board

    Returns:
        FEN string representing the puzzle
    """

    def format_list_to_string(input_list):
        return "/".join(
            "".join(str(item) for item in input_list[i : i + 8])
            for i in range(0, len(input_list), 8)
        )

    def compress_zeros(board_string):
        result = []
        zero_count = 0

        for char in board_string:
            if char == "0":
                zero_count += 1
            else:
                if zero_count > 0:
                    result.append(str(zero_count))
                    zero_count = 0
                result.append(char)

            if char == "/":
                if zero_count > 0:
                    result[-1] = str(zero_count)
                    zero_count = 0

        if zero_count > 0:
            result.append(str(zero_count))

        return "".join(result)

    # Randomly select positions
    positions = random.sample(range(64), num_pieces)

    # Assign pieces to positions
    chesses = [random.choice(piece_types) for _ in range(num_pieces)]

    # Create array
    arr = []
    index = 0
    for i in range(64):
        if i in positions:
            arr.append(chesses[index])
            index += 1
        else:
            arr.append(0)

    arr = format_list_to_string(arr)
    arr = compress_zeros(arr)

    # Verify puzzle is solvable
    puzzle = Board(arr)
    solver_puzzle = Solver(puzzle)

    if solver_puzzle.solve() is not None:
        return arr
    else:
        return generate_random_puzzle(piece_types, num_pieces)


# ============================================================================
# PIL Rendering
# ============================================================================


class ChessBoardImage:
    """Renders a chess board as a PIL Image."""

    def __init__(self, fen: str):
        self.board_size = 8
        self.cell_size = 60
        border_width = 10
        label_padding = 20
        font_size = 16

        try:
            self.font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            self.font = ImageFont.load_default()

        # Create image with space for borders and labels
        image_width = (
            (self.board_size * self.cell_size) + border_width * 2 + label_padding
        )
        image_height = (
            (self.board_size * self.cell_size) + border_width * 2 + label_padding
        )
        self.image = Image.new("RGB", (image_width, image_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.load_pieces()
        self.draw_board(border_width, label_padding)

        offset_x = border_width + label_padding // 2
        offset_y = border_width + label_padding // 2
        self.set_up_board(fen, offset_x, offset_y)

    def draw_board(self, border_width, label_padding):
        """Draw the chessboard grid and labels."""
        colors = ["white", "gray"]
        for row in range(self.board_size):
            for col in range(self.board_size):
                x1 = col * self.cell_size + border_width + label_padding // 2
                y1 = row * self.cell_size + border_width + label_padding // 2
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                color = colors[(row + col) % 2]
                self.draw.rectangle([x1, y1, x2, y2], fill=color)

        # Draw border
        self.draw.rectangle(
            [
                border_width + label_padding // 2 - 1,
                border_width + label_padding // 2 - 1,
                self.board_size * self.cell_size + border_width + label_padding // 2,
                self.board_size * self.cell_size + border_width + label_padding // 2,
            ],
            outline="black",
            width=2,
        )

        # Add row numbers (1-8 from bottom to top)
        for i in range(self.board_size):
            y = (
                self.board_size * self.cell_size
                - i * self.cell_size
                + border_width
                + label_padding // 2
                - self.cell_size // 2
            )
            text = str(i + 1)
            bbox = self.draw.textbbox((0, 0), text, font=self.font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            x = border_width - text_width // 2
            self.draw.text(
                (x, y - text_height // 2), text, fill="black", font=self.font
            )

        # Add column letters (a-h from left to right)
        for i in range(self.board_size):
            x = (
                i * self.cell_size
                + border_width
                + label_padding // 2
                + self.cell_size // 2
            )
            text = chr(ord("a") + i)
            bbox = self.draw.textbbox((0, 0), text, font=self.font)
            text_width = bbox[2] - bbox[0]
            y = self.board_size * self.cell_size + border_width + label_padding + 5
            self.draw.text(
                (x - text_width // 2, y),
                text,
                fill="black",
                font=self.font,
                anchor="ms",
            )

    def load_pieces(self):
        """Load piece images from disk."""
        pieces = {
            "R": "rook",
            "N": "knight",
            "B": "bishop",
            "Q": "queen",
            "K": "king",
            "P": "pawn",
        }
        self.piece_images = {}
        pieces_dir = Path(
            "/mnt/petrelfs/gujiawei/jiawei/env-v/Game-RL/src/chess_ranger/plot_image"
        )

        for key, value in pieces.items():
            image_path = pieces_dir / f"{value}.png"
            img = Image.open(image_path)
            img = img.resize((self.cell_size, self.cell_size), Image.LANCZOS)
            self.piece_images[key] = img

    def set_up_board(self, fen: str, offset_x: int, offset_y: int):
        """Place pieces on the board according to FEN notation."""
        rows = fen.split("/")
        board_setup = []
        for r, row in enumerate(rows):
            col = 0
            for piece in row:
                if piece.isdigit():
                    col += int(piece)
                elif piece.isalpha():
                    piece_type = piece.upper()
                    board_setup.append((piece_type, col, r))
                    col += 1

        for piece, col, row in board_setup:
            x = col * self.cell_size + offset_x
            y = row * self.cell_size + offset_y
            piece_image = self.piece_images[piece]
            self.image.paste(piece_image, (x, y), mask=piece_image)

    def get_image(self) -> Image.Image:
        """Return the rendered board image."""
        return self.image


# ============================================================================
# Helper Functions
# ============================================================================


PIECE_NAME_MAPPING = {
    "P": "Pawn",
    "R": "Rook",
    "N": "Knight",
    "B": "Bishop",
    "Q": "Queen",
    "K": "King",
}


def get_plot_level(num_pieces: int) -> str:
    """Determine difficulty based on number of pieces."""
    if num_pieces == 4:
        return "Easy"
    elif num_pieces == 5:
        return "Medium"
    elif num_pieces >= 6:
        return "Hard"
    else:
        return "Unknown"


def count_pieces_in_fen(fen: str) -> tuple[dict, dict]:
    """Count pieces and their positions in FEN notation.

    Returns:
        (piece_count, piece_positions) where:
        - piece_count: dict of {piece_symbol: count}
        - piece_positions: dict of {piece_symbol: [(row, col), ...]}
    """
    piece_count = {"P": 0, "R": 0, "N": 0, "B": 0, "Q": 0, "K": 0}
    piece_positions = {"P": [], "R": [], "N": [], "B": [], "Q": [], "K": []}

    rows = fen.split("/")
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            elif char.isalpha():
                piece_count[char.upper()] += 1
                piece_positions[char.upper()].append((row_idx, col_idx))
                col_idx += 1

    return piece_count, piece_positions


def convert_to_chess_notation(row_idx: int, col_idx: int) -> str:
    """Convert (row, col) to chess notation (e.g., (3, 2) -> 'c5')."""
    col = chr(ord("a") + col_idx)
    row = 8 - row_idx
    return f"{col}{row}"


# ============================================================================
# Chess Ranger QA Environment
# ============================================================================


class GameRLChessRangerQAEnv(Env):
    """Chess Ranger QA environment.

    This environment provides 5 question types:
    0. aq: How many steps needed to solve the puzzle? (Hard)
    1. count: How many [piece]s are on the board? (Medium)
    2. find: Where is the [piece] on the board? (Easy)
    3. pos: What piece is at [square]? (Easy)
    4. predict: Which moves lead to solvable states? (Hard)
    """

    QUESTION_TYPES = [
        {
            "id": "steps_to_solve",
            "name": "Steps to Solve",
            "level": "Hard",
            "answer_format": "fill_in_blank",
            "qa_type": "State Prediction",
            "question_id": 1,
        },
        {
            "id": "count_pieces",
            "name": "Count Pieces",
            "level": "Medium",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
            "question_id": 2,
        },
        {
            "id": "find_piece",
            "name": "Find Piece",
            "level": "Easy",
            "answer_format": "fill_in_blank",
            "qa_type": "Target Perception",
            "question_id": 3,
        },
        {
            "id": "piece_at_square",
            "name": "Piece at Square",
            "level": "Easy",
            "answer_format": "multiple_choice",
            "qa_type": "Target Perception",
            "question_id": 4,
        },
        {
            "id": "predict_solvable",
            "name": "Predict Solvable Moves",
            "level": "Hard",
            "answer_format": "multiple_choice",
            "qa_type": "State Prediction",
            "question_id": 5,
        },
    ]

    GAME_RULES = dedent("""
        This game is called Chess Ranger. The rules are as follows:
        - Pieces move like in standard chess
        - You can only perform capture moves
        - The king is allowed to be captured
        - The goal is to end up with a single piece remaining on the board
    """).strip()

    def __init__(self, num_pieces: int = 6, question_type: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.num_pieces = num_pieces
        self._question_type = question_type
        self._current_question = None
        self._fen = None
        self._board = None

    @property
    def description(self) -> str:
        return f"Chess Ranger QA\n\n{self.GAME_RULES}"

    def _get_state_text(self) -> str:
        """Generate text description of current chess board state."""
        pieces = self._board.get_piece_positions()

        text = "Chess Ranger Puzzle\n"
        text += "Board: 8x8 chess board\n"
        text += f"Pieces: {len(pieces)}\n\n"

        text += "Current Board Position:\n"
        for piece_name, position in pieces:
            text += f"  {piece_name} at {position}\n"

        return text.strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)

        # Generate puzzle
        piece_types = ["P", "R", "B", "N", "Q", "K"]
        self._fen = generate_random_puzzle(piece_types, self.num_pieces)
        self._board = Board(self._fen)

        # Select question type
        q_type = (
            self._question_type
            if self._question_type is not None
            else random.randint(0, 4)
        )

        # Generate question
        if q_type == 0:
            self._current_question = self._generate_steps_to_solve_question()
        elif q_type == 1:
            self._current_question = self._generate_count_pieces_question()
        elif q_type == 2:
            self._current_question = self._generate_find_piece_question()
        elif q_type == 3:
            self._current_question = self._generate_piece_at_square_question()
        elif q_type == 4:
            self._current_question = self._generate_predict_solvable_question()

        # Render board
        board_image = ChessBoardImage(self._fen)
        image = board_image.get_image()

        # Generate text state
        text_state = self._get_state_text()

        obs = Observation(
            image=image,
            text=text_state,
            metadata={
                "question": self._current_question["question"],
            },
        )

        info = {
            "oracle_answer": self._current_question["answer"],
            "question_type": self.QUESTION_TYPES[q_type]["id"],
        }

        return obs, info

    def inner_step(
        self, action: str
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        # Normalize answer
        answer_normalized = action.strip().lower()
        correct_answer = self._current_question["answer"].strip().lower()

        # Check if correct
        correct = answer_normalized == correct_answer
        reward = 1.0 if correct else 0.0

        # Generate response
        if correct:
            response = "Correct!"
        else:
            response = f"Incorrect. The correct answer is: {self._current_question['answer']}\n\n{self._current_question['analysis']}"

        # Re-render board for response
        board_image = ChessBoardImage(self._fen)
        image = board_image.get_image()

        obs = Observation(image=image, text=response)
        return obs, reward, True, False, {}

    def _generate_steps_to_solve_question(self) -> dict:
        """Type 0: How many steps are needed to solve the puzzle?"""
        # Solve the puzzle using a copy (solver modifies the board!)
        board_copy_for_solve = Board(self._fen)
        solver = Solver(board_copy_for_solve)
        moves = solver.solve()  # Returns list or None

        if moves is None:
            # Regenerate puzzle if not solvable (should not happen)
            piece_types = ["P", "R", "B", "N", "Q", "K"]
            self._fen = generate_random_puzzle(piece_types, self.num_pieces)
            self._board = Board(self._fen)
            return self._generate_steps_to_solve_question()

        num_steps = len(moves)  # Use length of moves list

        # Get current board situation (from original board, not modified one)
        pieces_positions = self._board.get_piece_positions()
        analysis = "The current board situation is: "
        for piece_name, position in pieces_positions:
            analysis += f"{piece_name} at {position}. "

        # Get detailed trace
        board_copy = Board(self._fen)
        trace_solver = TraceSolver(board_copy)
        _, trace = trace_solver.solve()

        analysis += "In the process of solving the puzzle, we have 2 operations and 2 flags to analyze: "
        analysis += "operation Try indicates the execution of a step, "
        analysis += "operation Backtrack indicates that the capture operation performed by the last Try step is withdrawn; "
        analysis += "flag Fail indicates all tries have failed in the current case, "
        analysis += "flag Success indicates that the puzzle is solved. "
        analysis += f"The solved steps are as follows: {' '.join(trace)} "
        analysis += f"So the total number of steps is {num_steps}."

        question = (
            f"{self.GAME_RULES}\n\nHow many steps are needed to solve the puzzle?"
        )

        return {"question": question, "answer": str(num_steps), "analysis": analysis}

    def _generate_count_pieces_question(self) -> dict:
        """Type 1: How many [piece]s are on the board?"""
        piece_types = ["P", "R", "B", "N", "Q", "K"]
        piece_count, piece_positions = count_pieces_in_fen(self._fen)

        # Randomly select a piece type
        random_piece_type = random.choice(piece_types)
        count = piece_count[random_piece_type]
        positions = piece_positions[random_piece_type]
        piece_full_name = PIECE_NAME_MAPPING[random_piece_type]

        # Build analysis
        pieces_board = self._board.get_piece_positions()
        analysis = "The current board situation is: "
        for piece_name, position in pieces_board:
            analysis += f"{piece_name} at {position}. "

        if count == 0:
            analysis += f"There's no {piece_full_name} on the board. So the number of {piece_full_name} is {count}."
        else:
            positions_str = ", ".join(
                [convert_to_chess_notation(r, c) for r, c in positions]
            )
            analysis += f"The {piece_full_name} is in the following positions on the board: {positions_str}. "
            analysis += f"So the number of {piece_full_name} is {count}."

        question = f"How many {piece_full_name}s are on the board?"

        return {"question": question, "answer": str(count), "analysis": analysis}

    def _generate_find_piece_question(self) -> dict:
        """Type 2: Where is the [piece] on the board?"""
        piece_types = ["P", "R", "B", "N", "Q", "K"]
        piece_count, piece_positions = count_pieces_in_fen(self._fen)

        # Find pieces with exactly 1 instance
        single_pieces = [p for p in piece_types if piece_count[p] == 1]

        if not single_pieces:
            # If no single pieces, regenerate puzzle
            piece_types_gen = ["P", "R", "B", "N", "Q", "K"]
            self._fen = generate_random_puzzle(piece_types_gen, self.num_pieces)
            self._board = Board(self._fen)
            return self._generate_find_piece_question()

        # Select a piece with exactly 1 instance
        random_piece_type = random.choice(single_pieces)
        positions = piece_positions[random_piece_type]
        piece_full_name = PIECE_NAME_MAPPING[random_piece_type]

        # Build analysis
        pieces_board = self._board.get_piece_positions()
        analysis = "The current board situation is: "
        for piece_name, position in pieces_board:
            analysis += f"{piece_name} at {position}. "

        positions_str = convert_to_chess_notation(positions[0][0], positions[0][1])
        analysis += f"There is exactly one {piece_full_name} on the board, located at position: {positions_str}."

        question = f"Where is the {piece_full_name} on the board?"

        return {"question": question, "answer": positions_str, "analysis": analysis}

    def _generate_piece_at_square_question(self) -> dict:
        """Type 3: What piece is at [square]?"""
        piece_count, piece_positions = count_pieces_in_fen(self._fen)

        # Randomly select a square
        while True:
            random_row = random.randint(0, 7)
            random_col = random.randint(0, 7)

            # Check if there's a piece at this position
            piece_at_position = "No Piece"
            for piece, positions in piece_positions.items():
                if (random_row, random_col) in positions:
                    piece_at_position = PIECE_NAME_MAPPING[piece]
                    break

            # Bias towards occupied squares (but allow empty sometimes)
            if piece_at_position != "No Piece" or random.random() < 0.3:
                break

        # Build analysis
        pieces_board = self._board.get_piece_positions()
        analysis = "The current board situation is: "
        for piece_name, position in pieces_board:
            analysis += f"{piece_name} at {position}. "

        position_str = convert_to_chess_notation(random_row, random_col)
        analysis += (
            f"The piece at {position_str} is {piece_at_position}. So the option is "
        )

        # Generate multiple choice options
        options = ["Pawn", "Rook", "Knight", "Bishop", "Queen", "King", "No Piece"]
        answer_index = options.index(piece_at_position)
        letters = ["A", "B", "C", "D", "E", "F", "G"]
        answer_letter = letters[answer_index]

        analysis += f"{answer_letter}."

        options_str = ", ".join(
            [f"{letters[i]}.{options[i]}" for i in range(len(options))]
        )
        question = f"What piece is at {position_str}? Choose from the following options: {options_str}"

        return {"question": question, "answer": answer_letter, "analysis": analysis}

    def _generate_predict_solvable_question(self) -> dict:
        """Type 4: Which of the following moves will lead to a state that can still be solved successfully?"""
        # Get all possible moves
        all_moves = self._board.moves()

        # Need at least 4 moves
        if len(all_moves) < 4:
            piece_types = ["P", "R", "B", "N", "Q", "K"]
            self._fen = generate_random_puzzle(piece_types, self.num_pieces)
            self._board = Board(self._fen)
            return self._generate_predict_solvable_question()

        # Evaluate each move
        all_steps = []
        for move in all_moves:
            board_next = Board(self._fen)
            board_next.make_move(move)

            next_solver = Solver(board_next)
            is_solvable = next_solver.solve() is not None

            all_steps.append(
                {
                    "move": f"move {move.src_piece} in {move.src_square} to capture {move.dst_piece} in {move.dst_square}",
                    "solvable": is_solvable,
                }
            )

        # Randomly select 4 moves
        random.shuffle(all_steps)
        selected_steps = all_steps[:4]

        # Build analysis
        pieces_board = self._board.get_piece_positions()
        analysis = "The current board situation is: "
        for piece_name, position in pieces_board:
            analysis += f"{piece_name} at {position}. "

        # Generate options and find correct answers
        options = [step["move"] for step in selected_steps]
        answer_indices = [
            i for i, step in enumerate(selected_steps) if step["solvable"]
        ]
        letters = ["A", "B", "C", "D"]

        if not answer_indices:
            # If no solvable moves, regenerate
            piece_types = ["P", "R", "B", "N", "Q", "K"]
            self._fen = generate_random_puzzle(piece_types, self.num_pieces)
            self._board = Board(self._fen)
            return self._generate_predict_solvable_question()

        answer_letters = "".join([letters[i] for i in sorted(answer_indices)])

        analysis += "After checking each move: "
        for i, step in enumerate(selected_steps):
            if step["solvable"]:
                analysis += f"Option {letters[i]} leads to a solvable state. "
            else:
                analysis += f"Option {letters[i]} leads to an unsolvable state. "

        analysis += f"So the correct options are: {answer_letters}."

        options_str = ", ".join(
            [f"{letters[i]}.{options[i]}" for i in range(len(options))]
        )
        question = (
            f"{self.GAME_RULES}\n\n"
            f"Which of the following moves will lead to a state that can still be solved successfully? "
            f"Choose from the following options: {options_str}"
        )

        return {"question": question, "answer": answer_letters, "analysis": analysis}

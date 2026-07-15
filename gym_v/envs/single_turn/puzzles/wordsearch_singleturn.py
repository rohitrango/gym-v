"""Word Search single-turn environment: find all hidden words in one response."""

from __future__ import annotations

from importlib import resources
import re
import string
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation, get_logger

logger = get_logger()

GAME_RULES = dedent("""
    This is a Word Search puzzle game.

    Rules:
    1. The grid contains uppercase letters arranged in rows and columns
    2. Words can be placed in 8 directions: right, down, diagonal-right-down, diagonal-right-up, diagonal-left-down, diagonal-left-up, up, or left
    3. Row and column indexes begin from 1 at the top-left corner
    4. Words read from start to end in the specified direction
""").strip()


class WordSearchSingleTurnEnv(Env):
    # Meta: source=GameRL, category=puzzles, turn=single
    """Word Search single-turn environment.

    Hides multiple words in a grid and asks the agent to locate all of them
    in a single response. Each word answer is given as [r1 r2 c1 c2] where
    r1/c1 are start row/col and r2/c2 are end row/col (1-indexed).

    Args:
        num_words: Number of words to hide in the grid (default 4).
        grid_size: Grid size (default None for random 5-8).
        cell_size: Cell size in pixels for rendering (default 50).
    """

    DIRECTIONS = {
        "right": (0, 1),
        "down": (1, 0),
        "diagonal-right-down": (1, 1),
        "diagonal-right-up": (-1, 1),
        "diagonal-left-down": (1, -1),
        "diagonal-left-up": (-1, -1),
        "up": (-1, 0),
        "left": (0, -1),
    }

    gamerl_assets_dir = resources.files("gym_v.envs") / "assets" / "gamerl"
    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        num_words: int = 4,
        grid_size: int | None = None,
        cell_size: int = 50,
        num_players: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_words_param = num_words
        self._grid_size_param = grid_size
        self._cell_size = cell_size
        self._margin = 0
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        try:
            with open(self.gamerl_assets_dir / "words.txt") as f:
                self._words = [word.strip().upper() for word in f if word.strip()]
        except FileNotFoundError:
            self._words = [
                "PYTHON", "CODE", "DATA", "WORD", "SEARCH", "GRID",
                "LETTER", "PUZZLE", "GAME", "FIND", "DIRECTION", "RANDOM", "MATRIX",
            ]

        self._grid: list[list[str]] = []
        self._grid_size: int = 5

        # Placed word state: word -> (r1, c1, r2, c2) all 1-indexed
        self._placed_words: list[str] = []
        self._placed_positions: dict[str, tuple[int, int, int, int]] = {}

    @property
    def description(self) -> str:
        words_list = ", ".join(self._placed_words)
        num = len(self._placed_words)
        return dedent(f"""
            {GAME_RULES}

            Find the following {num} word(s) hidden in the grid: {words_list}

            For each word, output its location as [r1 c1 r2 c2] where:
            - r1 = start row (1-indexed from top)
            - c1 = start column (1-indexed from left)
            - r2 = end row
            - c2 = end column

            Submit all answers separated by spaces, one bracket per word. Example:
            [2 1 2 5] [4 3 7 3] [1 3 1 6]

            You must find all {num} words. Longer words correctly found contribute more to your score.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        if self._grid_size_param is not None:
            self._grid_size = self._grid_size_param
        else:
            self._grid_size = self.py_random.randint(5, 8)

        self._generate_grid()
        self._place_words()

        logger.info(
            f"Reset Word Search single-turn ({self._grid_size}x{self._grid_size}, "
            f"{len(self._placed_words)} words: {self._placed_words})."
        )

        text_prompt = self.description
        obs = Observation(
            image=self.render(),
            text=text_prompt,
            metadata={
                "text_prompt": text_prompt,
                "words": self._placed_words,
                "positions": self._placed_positions,
            },
        )
        info = {
            "words": self._placed_words,
            "positions": self._placed_positions,
        }
        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]

        guesses = self._parse_guesses(action_str)
        found_words: list[str] = []

        for r1, c1, r2, c2 in guesses:
            word = self._word_at(r1, c1, r2, c2)
            if word and word not in found_words:
                found_words.append(word)

        total_letters = sum(len(w) for w in self._placed_words)
        found_letters = sum(len(w) for w in found_words)
        reward = found_letters / total_letters if total_letters > 0 else 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "text_prompt": self.description,
                "words": self._placed_words,
                "found_words": found_words,
            },
        )
        info = {
            "words": self._placed_words,
            "found_words": found_words,
            "positions": self._placed_positions,
            "guesses": guesses,
            "found_letters": found_letters,
            "total_letters": total_letters,
        }

        terminated = True
        truncated = False

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
        grid_width = (self._grid_size + 1) * self._cell_size
        grid_height = (self._grid_size + 1) * self._cell_size
        img_width = grid_width + self._margin
        img_height = grid_height + self._margin

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype(str(self.assets_dir / "DejaVuSans.ttf"), 32)
        except Exception:
            font = ImageFont.load_default()

        for i in range(self._grid_size + 1):
            draw.line(
                [(i * self._cell_size, 0), (i * self._cell_size, grid_height)],
                fill=(0, 0, 0),
                width=1,
            )
            draw.line(
                [(0, i * self._cell_size), (grid_width, i * self._cell_size)],
                fill=(0, 0, 0),
                width=1,
            )

        for i in range(self._grid_size):
            draw.text(
                (5, (i + 1) * self._cell_size + 5),
                str(i + 1),
                fill=(0, 0, 0),
                font=font,
            )

        for j in range(self._grid_size):
            draw.text(
                ((j + 1) * self._cell_size + 15, 5),
                str(j + 1),
                fill=(0, 0, 0),
                font=font,
            )

        for i in range(self._grid_size):
            for j in range(self._grid_size):
                draw.text(
                    ((j + 1) * self._cell_size + 15, (i + 1) * self._cell_size + 5),
                    self._grid[i][j],
                    fill=(0, 0, 0),
                    font=font,
                )

        return img

    # ------------------------------------------------------------------
    # Grid generation (from word_search.py)
    # ------------------------------------------------------------------

    def _generate_grid(self) -> None:
        self._grid = [
            [self.py_random.choice(string.ascii_uppercase) for _ in range(self._grid_size)]
            for _ in range(self._grid_size)
        ]

    def _insert_word(
        self, word: str, start_row: int, start_col: int, direction: tuple[int, int]
    ) -> bool:
        curr_row, curr_col = start_row, start_col
        for letter in word:
            if 0 <= curr_row < self._grid_size and 0 <= curr_col < self._grid_size:
                self._grid[curr_row][curr_col] = letter
                curr_row += direction[0]
                curr_col += direction[1]
            else:
                return False
        return True

    def _can_place_word(
        self, word: str, start_row: int, start_col: int, direction: tuple[int, int]
    ) -> bool:
        curr_row, curr_col = start_row, start_col
        for _ in word:
            if not (0 <= curr_row < self._grid_size and 0 <= curr_col < self._grid_size):
                return False
            curr_row += direction[0]
            curr_col += direction[1]
        return True

    def _place_words(self) -> None:
        self._placed_words = []
        self._placed_positions = {}

        candidates = [w for w in self._words if 3 <= len(w) <= self._grid_size]
        self.py_random.shuffle(candidates)

        for word in candidates:
            if len(self._placed_words) >= self._num_words_param:
                break

            placed = False
            directions = list(self.DIRECTIONS.items())
            self.py_random.shuffle(directions)

            for dir_name, direction in directions:
                if placed:
                    break
                positions = [
                    (r, c)
                    for r in range(self._grid_size)
                    for c in range(self._grid_size)
                ]
                self.py_random.shuffle(positions)
                for start_row, start_col in positions:
                    if self._can_place_word(word, start_row, start_col, direction):
                        self._insert_word(word, start_row, start_col, direction)
                        dr, dc = direction
                        end_row = start_row + dr * (len(word) - 1)
                        end_col = start_col + dc * (len(word) - 1)
                        # Store 1-indexed (r1, c1, r2, c2)
                        self._placed_positions[word] = (
                            start_row + 1, start_col + 1,
                            end_row + 1, end_col + 1,
                        )
                        self._placed_words.append(word)
                        placed = True
                        break

    # ------------------------------------------------------------------
    # Answer parsing and checking
    # ------------------------------------------------------------------

    def _parse_guesses(self, action_str: str) -> list[tuple[int, int, int, int]]:
        """Extract all [r1 c1 r2 c2] bracket entries from the action string."""
        matches = re.findall(r'\[\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*\]', action_str)
        return [(int(a), int(b), int(c), int(d)) for a, b, c, d in matches]

    def _word_at(self, r1: int, c1: int, r2: int, c2: int) -> str | None:
        """Return the placed word whose start/end position matches, or None."""
        for word, (wr1, wc1, wr2, wc2) in self._placed_positions.items():
            if wr1 == r1 and wc1 == c1 and wr2 == r2 and wc2 == c2:
                return word
        return None

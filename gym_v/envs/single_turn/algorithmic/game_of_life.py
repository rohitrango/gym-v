"""Game of Life single-turn environment backed by reasoning-gym."""

from __future__ import annotations

from importlib import resources
import json
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw
from reasoning_gym.factory import create_dataset

from gym_v import Env, Observation, get_logger

logger = get_logger()


class GameOfLifeEnv(Env):
    # Meta: source=ReasoningGym, category=algorithmic, turn=single
    """Conway's Game of Life simulation using reasoning-gym's dataset.

    The player predicts the state of the board after N simulation steps.
    This tests understanding of cellular automaton rules.

    Args:
        dataset_kwargs: Configuration parameters for the reasoning-gym dataset
        cell_px: Size of each cell in pixels for rendering
        padding: Padding around the grid in pixels for rendering
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    def __init__(
        self,
        dataset_kwargs: dict[str, Any] | None = None,
        cell_px: int = 32,
        padding: int = 16,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._dataset_kwargs = dataset_kwargs or {}
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._dataset = None
        self._entry: dict[str, Any] | None = None
        self._entry_idx: int | None = None
        self._metadata: dict[str, Any] | None = None
        self._board: list[list[int]] | None = None
        self._oracle_answer: str | None = None

    @property
    def description(self) -> str:
        """Return description with current simulation parameters.

        Original reasoning-gym question format:
        ```
        What will this Game of Life board look like after {simulation_steps} steps of simulation?
        Assume a Moore neighborhood and wrapping topology.
        Reply as array of arrays representing rows in the grid from top to bottom in JSON format.
        (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

        [[0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,1,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0],
         ...
         [0,0,0,0,0,0,0,0,0,0]].
        ```

        Original reasoning-gym answer format:
        ```
        [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],...]
        ```
        (JSON array of arrays, each inner array is a row, 0=dead, 1=alive)
        """
        sim_steps = self._metadata.get("simulation_steps", 1) if self._metadata else 1
        rows = len(self._board) if self._board else 0
        cols = len(self._board[0]) if self._board and self._board[0] else 0

        return dedent(f"""
            What will this Game of Life board look like after {sim_steps} step(s) of simulation?
            Assume a Moore neighborhood and wrapping topology.
            Reply as array of arrays representing rows in the grid from top to bottom in JSON format.
            (An empty 3x3 grid would look like this: [[0,0,0],[0,0,0],[0,0,0]])

            In the image:
            - Dark/black cells are alive (1)
            - Light/gray cells are dead (0)
            - The grid is {rows}x{cols}
        """).strip()

    def _make_dataset(self, *, seed: int | None):
        kwargs = self._dataset_kwargs.copy()
        if "seed" not in kwargs:
            kwargs["seed"] = seed if seed is not None else int(self.np_random.integers(0, 2**31))
        if "size" not in kwargs:
            kwargs["size"] = 500

        return create_dataset("game_of_life", **kwargs)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed

        self._dataset = self._make_dataset(seed=self._seed)
        self._entry_idx = int(self.np_random.integers(0, len(self._dataset)))
        self._entry = self._dataset[self._entry_idx]

        self._oracle_answer = self._entry["answer"]
        self._metadata = self._entry.get("metadata", {})

        # Extract the initial board from question
        self._board = self._parse_board_from_question(self._entry["question"])

        logger.info("Reset ReasoningGym Game of Life.")

        # obs.text = only the board state as JSON (caption), not the full question
        board_text = self._board_to_json(self._board) if self._board else "[]"

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                "state_text": board_text,
                **self._metadata,
                "text_prompt": self._entry.get("question", ""),
            },
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
        }
        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def _parse_board_from_question(self, question: str) -> list[list[int]]:
        """Parse the board JSON from the question string.

        The question contains an example like [[0,0,0],[0,0,0],[0,0,0]] and
        the actual board. We need to find the last/largest board array.
        """
        try:
            # Find the last occurrence of [[ which should be the actual board
            # (the first one is typically an example in the prompt)
            last_start = question.rfind("[[")
            end = question.rfind("]]") + 2
            if last_start != -1 and end > last_start:
                board_str = question[last_start:end]
                # Normalize whitespace/newlines for JSON parsing
                board_str = board_str.replace("\n", "").replace(" ", "")
                return json.loads(board_str)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse board JSON: {e}")
        return []

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
        answer = action[agent_id]
        try:
            reward = self._dataset.score_answer(answer=answer, entry=self._entry)
        except Exception as e:
            logger.warning(f"score_answer failed for {type(self).__name__}: {e}, answer={str(answer)[:200]}")
            reward = 0.0

        obs = Observation(
            image=self.render(),
            text=None,
            metadata={
                **self._metadata,
                "text_prompt": self._entry.get("question", ""),
            },
        )
        info = {
            "reasoning_gym_seed": self._seed,
            "reasoning_gym_index": self._entry_idx,
            "oracle_answer": self._oracle_answer,
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
        return self._render_life_grid(
            self._board, cell_px=self._cell_px, padding=self._padding
        )

    def _render_life_grid(
        self,
        board: list[list[int]],
        cell_px: int = 32,
        padding: int = 16,
        bg: tuple[int, int, int] = (250, 250, 250),
        dead_color: tuple[int, int, int] = (240, 240, 240),
        alive_color: tuple[int, int, int] = (50, 50, 50),
        grid_color: tuple[int, int, int] = (200, 200, 200),
    ) -> Image.Image:
        if not board:
            return Image.new("RGB", (200, 200), bg)

        rows = len(board)
        cols = len(board[0]) if board else 0
        width = padding * 2 + cell_px * cols
        height = padding * 2 + cell_px * rows
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        # Draw cells
        for r in range(rows):
            for c in range(cols):
                x = padding + c * cell_px
                y = padding + r * cell_px

                cell_val = board[r][c] if r < len(board) and c < len(board[r]) else 0
                color = alive_color if cell_val == 1 else dead_color

                draw.rectangle(
                    [x, y, x + cell_px - 1, y + cell_px - 1],
                    fill=color,
                    outline=grid_color,
                    width=1,
                )

        # Draw outer border
        draw.rectangle(
            [padding - 1, padding - 1, width - padding, height - padding],
            outline=(100, 100, 100),
            width=2,
        )

        return img

    def _board_to_json(self, board: list[list[int]]) -> str:
        """Convert board to JSON string representation."""
        rows = [json.dumps(row, separators=(",", ":")) for row in board]
        return "[" + ",\n ".join(rows) + "]"

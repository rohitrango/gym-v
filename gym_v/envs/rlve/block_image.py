"""Block image environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEBlockImageEnv(Env):
    """RLVE Block Image as a single-turn environment.

    This environment generates 3D isometric projections of stacked cubes
    on a grid. Given a matrix where each cell contains a number indicating
    the height of cubes stacked at that position, the task is to output
    an ASCII-art representation of the 3D structure.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a {M} × {N} rectangular grid, where each cell represents a stack of identical cube blocks. Each cube has size 1 × 1 × 1, and no rotation or flipping is allowed — all cubes are placed in the same orientation.
You are given a matrix representing the number of cubes stacked on each cell in the grid (the integer at row i and column j indicates how many cube blocks are stacked on the cell located at row i, column j):
{matrix}

The visual representation of a **single cube** follows this fixed format:

$$
\def\arraystretch{{1e-10}}
\begin{{{{aligned}}}}
&\verb!  +---+!\\
&\verb! /   /|!\\
&\verb!+---+ |!\quad\textsf{{{{height}}}}\\
&\verb!|   | +!\\
&\verb!|   |/ !\quad\textsf{{{{width}}}}\\
&\verb!+---+  !\\
& \quad\textsf{{{{length}}}}
\end{{{{aligned}}}}
$$

Each `+` represents a corner, `-` spans the cube's length, `/` shows depth (width), and `|` shows height. Empty space in the final drawing should be represented using `.`.

The 3D isometric projection follows specific stacking rules:

- **Two cubes side by side (left/right):**
$$
\def\arraystretch{{1e-10}}
\begin{{{{aligned}}}}
\verb!..+---+---+!\\
\verb!./   /   /|!\\
\verb!+---+---+ |!\\
\verb!|   |   | +!\\
\verb!|   |   |/.!\\
\verb!+---+---+..!\\
\end{{{{aligned}}}}
$$

- **Two cubes stacked vertically (top/bottom):**
$$
\def\arraystretch{{1e-10}}
\begin{{{{aligned}}}}
\verb!..+---+!\\
\verb!./   /|!\\
\verb!+---+ |!\\
\verb!|   | +!\\
\verb!|   |/|!\\
\verb!+---+ |!\\
\verb!|   | +!\\
\verb!|   |/.!\\
\verb!+---+..!\\
\end{{{{aligned}}}}
$$

- **Two cubes front/back (depth):**
$$
\def\arraystretch{{1e-10}}
\begin{{{{aligned}}}}
\verb!....+---+!\\
\verb!.../   /|!\\
\verb!..+---+ |!\\
\verb!./   /| +!\\
\verb!+---+ |/.!\\
\verb!|   | +..!\\
\verb!|   |/...!\\
\verb!+---+....!\\
\end{{{{aligned}}}}
$$

The bottom-left corner of the lowest cube in cell ({M}, 1) (bottom row, first column) should align with the bottom-left of the entire drawing.

**Output Format:**
Your final output should be a string matrix of dimensions K × L (i.e., it has K lines separated by line breaks, with each line containing exactly L characters), where K is the number of rows and L is the number of columns **required to draw the 3D structure correctly** according to the rules above.

---

**Example 1**

When the rectangular grid is 1 × 2, and the number of cubes in each cell is as follows:
1 3

The output is (do **NOT** include the backticks or quotes — use the format below exactly):
```
......+---+
...../   /|
....+---+ |
....|   | +
....|   |/|
....+---+ |
..+-|   | +
./  |   |/|
+---+---+ |
|   |   | +
|   |   |/.
+---+---+..
```

---

**Example 2**

When the rectangular grid is 3 × 4, and the number of cubes in each cell is as follows:
2 2 1 2
2 2 1 1
3 2 1 2

The output is (do **NOT** include the backticks or quotes — use the format below exactly):
```
......+---+---+...+---+
..+---+  /   /|../   /|
./   /|-+---+ |.+---+ |
+---+ |/   /| +-|   | +
|   | +---+ |/+---+ |/|
|   |/   /| +/   /|-+ |
+---+---+ |/+---+ |/| +
|   |   | +-|   | + |/.
|   |   |/  |   |/| +..
+---+---+---+---+ |/...
|   |   |   |   | +....
|   |   |   |   |/.....
+---+---+---+---+......
```
"""

    def __init__(
        self,
        max_m_n: int = 4,
        max_height: int = 5,
        cell_px: int = 48,
        padding: int = 32,
        num_players: int = 1,
        wrong_format: float = -1.0,
        invalid_answer: float = -0.5,
        wrong_size: float = 0.0,
        rewarding_strategy: str = "mean([gold=answer])^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 2.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_m_n = max_m_n
        self._max_height = max_height
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._rewards = {
            "wrong_format": wrong_format,
            "invalid_answer": invalid_answer,
            "wrong_size": wrong_size,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }

        self._M: int | None = None
        self._N: int | None = None
        self._grid: list[list[int]] | None = None
        self._prompt: str | None = None
        self._reference_answer: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._grid and self._M and self._N:
            size_hint = f"{self._M} x {self._N}"
        else:
            size_hint = "M x N"
        return dedent(
            f"""
            Block Image 3D Projection:

            Given a {size_hint} grid where each cell contains a number representing
            the height of stacked cubes at that position, output an ASCII-art
            isometric projection of the 3D structure.

            Rules:
            - Each cube is drawn with edges using +, -, /, and | characters
            - Cubes stack vertically (increasing height at same position)
            - Cubes align in rows (left/right) and columns (front/back)
            - Empty space is represented with '.'
            - The projection uses specific offsets for proper 3D appearance

            Output format:
            - Multiple lines of equal length
            - Each line contains only: . + - / | (and space)
            - Exact dimensions depend on grid size and cube heights
            - Bottom-left cube of position (M,1) aligns with drawing origin
            """
        ).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        obs = Observation(
            image=self._last_image,
            text=self._prompt,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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
        reward = float(self._score_answer(action_str))
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "rlve_prompt": self._prompt,
                "rlve_reference_answer": self._reference_answer,
            },
        )
        info = {
            "reference_answer": self._reference_answer,
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

    def _generate(self) -> None:
        """Generate a random block image puzzle."""
        if self._max_m_n < 1:
            raise ValueError("max_m_n must be >= 1")

        self._M = int(self.np_random.integers(1, self._max_m_n + 1))
        self._N = int(self.np_random.integers(1, self._max_m_n + 1))
        self._grid = [
            [
                int(self.np_random.integers(1, self._max_height + 1))
                for _ in range(self._N)
            ]
            for _ in range(self._M)
        ]

        # Calculate canvas dimensions
        max_row = 0
        max_col = 0
        for i in range(self._M):
            for j in range(self._N):
                a = self._grid[i][j]
                t = self._M - i - 1
                cand_col = 2 * t + 4 * j + 6
                if cand_col > max_col:
                    max_col = cand_col
                cand_row = 2 * t + 3 * (a - 1) + 5
                if cand_row > max_row:
                    max_row = cand_row

        height = max_row + 1
        width = max_col + 1
        canvas = [["." for _ in range(width)] for _ in range(height)]

        # Template for a single cube
        template = [
            "..+---+",
            "./   /|",
            "+---+ |",
            "|   | +",
            "|   |/.",
            "+---+..",
        ]

        # Draw all cubes
        for i in range(self._M):
            for j in range(self._N):
                a = self._grid[i][j]
                t = self._M - i - 1
                for k in range(a):
                    x_offset = 2 * t + 4 * j
                    y_offset = 2 * t + 3 * k
                    for r in range(6):
                        for c in range(7):
                            ch = template[r][c]
                            if ch != ".":
                                row_index = y_offset + (5 - r)
                                col_index = x_offset + c
                                canvas[row_index][col_index] = ch

        # Convert canvas to string (flip vertically for correct orientation)
        output_lines = []
        for row in range(height - 1, -1, -1):
            output_lines.append("".join(canvas[row]))
        self._reference_answer = "\n".join(output_lines)

    def _prompt_generate(self) -> str:
        """Generate the prompt text."""
        if self._grid is None or self._M is None or self._N is None:
            raise RuntimeError("No grid generated")
        return self.prompt_template.format(
            M=self._M,
            N=self._N,
            matrix="\n".join(" ".join(map(str, row)) for row in self._grid),
        )

    def _process(self, answer: str | None) -> list[str] | None:
        """Process the answer string into a list of lines."""
        if answer is None:
            return None
        answer = answer.strip()
        image = []
        for line in answer.splitlines():
            line = line.strip()
            if line:
                image.append(line)
        return image if image else None

    def _score_answer(self, answer: str) -> float:
        """Score the answer against the reference answer."""
        processed_result = self._process(answer)
        if processed_result is None:
            return self._rewards["wrong_format"]

        image = processed_result

        if not image:
            return self._rewards["wrong_format"]

        # Check that all rows have the same length
        for row in image:
            if len(row) != len(image[0]):
                return self._rewards["wrong_format"]
            # Check valid characters
            if not all(ch in ".+-/| " for ch in row):
                return self._rewards["invalid_answer"]

        # Check dimensions
        gold_image = self._reference_answer.split("\n")
        if len(image) != len(gold_image):
            return self._rewards["wrong_size"]
        if len(image[0]) != len(gold_image[0]):
            return self._rewards["wrong_size"]

        # Calculate character-wise accuracy
        total_correct = 0
        for gold_row, row in zip(gold_image, image):
            assert len(gold_row) == len(row)
            total_correct += sum(gold_row[i] == row[i] for i in range(len(gold_row)))
        total_cells = len(gold_image) * len(gold_image[0])

        # Apply rewarding strategy
        if self._rewards["rewarding_strategy"] == "mean([gold=answer])^beta":
            return self._rewards["rewarding_weight"] * (
                (total_correct / total_cells) ** self._rewards["rewarding_beta"]
            )
        elif self._rewards["rewarding_strategy"] == "gold=answer":
            return self._rewards["rewarding_weight"] * (total_correct == total_cells)
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self._rewards['rewarding_strategy']}"
            )

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render a beautiful 3D visualization of the block structure."""
        if self._grid is None or self._M is None or self._N is None:
            raise RuntimeError("No grid generated")

        # Calculate image dimensions based on 3D projection
        # Use isometric projection: x_screen = x - z, y_screen = y + 0.5*x + 0.5*z
        cube_size = self._cell_px
        padding = self._padding

        # Calculate bounding box for all cubes
        max_x_screen = 0
        max_y_screen = 0
        min_x_screen = 0
        min_y_screen = 0

        for i in range(self._M):
            for j in range(self._N):
                height = self._grid[i][j]
                # Position in grid coordinates (row i, col j)
                # x = column, z = row, y = height
                x, z = j, i
                for y in range(height + 1):
                    x_screen = (x - z) * cube_size
                    y_screen = -y * cube_size + (x + z) * cube_size * 0.5
                    max_x_screen = max(max_x_screen, x_screen)
                    max_y_screen = max(max_y_screen, y_screen)
                    min_x_screen = min(min_x_screen, x_screen)
                    min_y_screen = min(min_y_screen, y_screen)

        width = int(max_x_screen - min_x_screen) + padding * 2 + cube_size
        height = int(max_y_screen - min_y_screen) + padding * 2 + cube_size

        img = Image.new("RGB", (width, height), (245, 245, 250))
        draw = ImageDraw.Draw(img)

        # Offset to move origin
        offset_x = -min_x_screen + padding
        offset_y = height - padding

        # Function to convert 3D coordinates to screen coordinates
        def to_screen(x: float, z: float, y: float) -> tuple[float, float]:
            x_screen = (x - z) * cube_size + offset_x
            y_screen = -(-y * cube_size + (x + z) * cube_size * 0.5) + offset_y
            return (x_screen, y_screen)

        # Draw cubes in back-to-front order for proper occlusion
        # Sort by rendering order (back to front, bottom to top)
        cubes_to_draw = []
        for i in range(self._M):
            for j in range(self._N):
                height = self._grid[i][j]
                for k in range(height):
                    # Position: x=column, z=row, y=height
                    cubes_to_draw.append((j, i, k))

        # Sort by rendering order: larger z+x first (back), then smaller y (bottom)
        cubes_to_draw.sort(key=lambda pos: (-pos[1] - pos[0], pos[2]))

        # Color scheme for 3D depth
        for x, z, y in cubes_to_draw:
            # Define the 8 corners of the cube
            corners = [
                to_screen(x, z, y),  # bottom-front-left
                to_screen(x + 1, z, y),  # bottom-front-right
                to_screen(x + 1, z + 1, y),  # bottom-back-right
                to_screen(x, z + 1, y),  # bottom-back-left
                to_screen(x, z, y + 1),  # top-front-left
                to_screen(x + 1, z, y + 1),  # top-front-right
                to_screen(x + 1, z + 1, y + 1),  # top-back-right
                to_screen(x, z + 1, y + 1),  # top-back-left
            ]

            # Draw three visible faces
            # Top face (brightest)
            top_face = [corners[4], corners[5], corners[6], corners[7]]
            draw.polygon(top_face, fill=(220, 220, 240), outline=(60, 60, 80))

            # Left face (medium shade)
            left_face = [corners[0], corners[3], corners[7], corners[4]]
            draw.polygon(left_face, fill=(180, 180, 200), outline=(60, 60, 80))

            # Right face (darkest visible face)
            right_face = [corners[0], corners[1], corners[5], corners[4]]
            draw.polygon(right_face, fill=(140, 140, 170), outline=(60, 60, 80))

        # Draw grid labels
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            try:
                font = ImageFont.truetype(font_path, int(cube_size * 0.25))
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Add height labels on visible cubes
        for i in range(self._M):
            for j in range(self._N):
                height = self._grid[i][j]
                if height > 0:
                    # Label position at the top of the stack
                    x, z, y = j + 0.5, i + 0.5, height
                    label_pos = to_screen(x, z, y)
                    label_text = str(height)

                    # Get text bounding box for centering
                    bbox = draw.textbbox((0, 0), label_text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

                    draw.text(
                        (label_pos[0] - tw / 2, label_pos[1] - th / 2 - cube_size * 0.3),
                        label_text,
                        fill=(40, 40, 60),
                        font=font,
                    )

        return img

"""Maximum independent set grid environment for gym-v (self-contained)."""

from __future__ import annotations

from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMaximumIndependentSetGridEnv(Env):
    """RLVE Maximum Independent Set Grid as a single-turn environment."""

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given a matrix of size {N} × {M}. Select some cells such that **no two selected cells are adjacent** (i.e., no two selected cells share a horizontal or vertical edge). Try your best to maximize the sum of the values in the selected cells. The matrix is given below (in **row-major order**):
{matrix}

**Output Format:** Output {N} lines, each with {M} digits (0 or 1) and no separators. A `1` means the corresponding cell is selected; a `0` means it is not."""

    def __init__(
        self,
        max_n_m: int = 4,
        cell_px: int = 64,
        padding: int = 24,
        num_players: int = 1,
        rewarding_strategy: str = "(answer/gold)^beta",
        rewarding_weight: float = 1.0,
        rewarding_beta: float = 3.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_n_m = max_n_m
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._rewarding_strategy = rewarding_strategy
        self._rewarding_weight = rewarding_weight
        self._rewarding_beta = rewarding_beta

        self._matrix: list[list[int]] | None = None
        self._gold_answer: float | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._matrix:
            rows = len(self._matrix)
            cols = len(self._matrix[0]) if self._matrix[0] else 0
            size_hint = f"{rows} x {cols}"
        else:
            size_hint = "N x M"
        return dedent(
            f"""
            Maximum Independent Set Grid rules:
            1) Select cells to maximize the sum of their values.
            2) No two selected cells can be adjacent (horizontally or vertically).
            3) Each cell contains a positive integer value.

            In the image:
            - Each cell shows its value as an integer
            - The grid is {size_hint}
            - Goal: Select non-adjacent cells to maximize total value

            Output format: N lines with M digits (0 or 1), no separators.
            - '1' means the cell is selected
            - '0' means the cell is not selected
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return the text representation of the current state."""
        return self._prompt or ""

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)

        self._generate()
        self._prompt = self._prompt_generate()
        self._last_image = self.render()

        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
                "rlve_gold_answer": self._gold_answer,
            },
        )
        info = {
            "gold_answer": self._gold_answer,
            "oracle_answer": self._oracle_answer,
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
        state_text = self._get_state_text()
        obs = Observation(
            image=self._last_image,
            text=None,
            metadata={
                "state_text": state_text,
                "text_prompt": self._prompt,
                "rlve_gold_answer": self._gold_answer,
            },
        )
        info = {
            "gold_answer": self._gold_answer,
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

    def _generate(self) -> None:
        """Generate a maximum independent set grid problem.

        Uses a bipartite graph min-cut formulation to compute the optimal
        maximum weight independent set on a grid graph.
        """
        import networkx as nx

        max_n_m = self._max_n_m
        if max_n_m < 2:
            raise ValueError("max_n_m must be >= 2")

        N = int(self.np_random.integers(2, max_n_m + 1))
        M = int(self.np_random.integers(2, max_n_m + 1))

        # Generate random matrix with values between 1 and max(N, M)
        self._matrix = [
            [int(self.np_random.integers(1, max(N, M) + 1)) for _ in range(M)]
            for _ in range(N)
        ]

        # Total sum of all cell weights
        TOTAL = sum(sum(row) for row in self._matrix)
        # Use TOTAL as the "infinite" capacity for inter-cell edges
        INF = TOTAL

        # Build a directed graph for the min-cut formulation
        G = nx.DiGraph()
        SOURCE, SINK = "s", "t"

        # Add edges from SOURCE→odd‐parity cells and even‐parity cells→SINK
        # plus infinite‐capacity edges between adjacent cells
        for i in range(N):
            for j in range(M):
                u = (i, j)
                weight = self._matrix[i][j]

                if (i + j) % 2 == 1:
                    # Odd parity: source → u with capacity = weight
                    G.add_edge(SOURCE, u, capacity=weight)
                    # Connect to each of its neighbors with infinite capacity
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M:
                            v = (ni, nj)
                            G.add_edge(u, v, capacity=INF)
                else:
                    # Even parity: u → sink with capacity = weight
                    G.add_edge(u, SINK, capacity=weight)

        # Compute the maximum flow (which equals the minimum cut capacity)
        flow_value, flow_dict = nx.maximum_flow(G, SOURCE, SINK)

        # By König's theorem on bipartite graphs:
        # max_weight_independent_set = TOTAL - min_vertex_cover_weight
        # and min_vertex_cover_weight = flow_value
        self._gold_answer = TOTAL - flow_value
        if self._gold_answer <= 0:
            raise ValueError("Gold answer must be positive")

        # Compute the min-cut to find which cells are in the independent set
        # Build residual graph (forward edges with residual capacity + backward edges with flow)
        residual_G = nx.DiGraph()
        for u in G:
            for v in G[u]:
                capacity = G[u][v]["capacity"]
                flow = flow_dict[u].get(v, 0)
                residual = capacity - flow
                if residual > 0:
                    residual_G.add_edge(u, v)
                if flow > 0:
                    residual_G.add_edge(v, u)  # backward edge

        # Find the reachable set from SOURCE in the residual graph
        reachable = nx.descendants(residual_G, SOURCE)
        reachable.add(SOURCE)

        # The minimum vertex cover consists of:
        # - Odd parity cells NOT reachable from SOURCE
        # - Even parity cells that ARE reachable from SOURCE
        # The maximum independent set is the complement of the minimum vertex cover
        reference = []
        for i in range(N):
            row = []
            for j in range(M):
                cell = (i, j)
                if (i + j) % 2 == 1:
                    # Odd parity: in independent set if reachable from SOURCE
                    if cell in reachable:
                        row.append("1")
                    else:
                        row.append("0")
                else:
                    # Even parity: in independent set if NOT reachable from SOURCE
                    if cell not in reachable:
                        row.append("1")
                    else:
                        row.append("0")
            reference.append("".join(row))

        self._oracle_answer = "\n".join(reference)

    def _prompt_generate(self) -> str:
        if self._matrix is None:
            raise RuntimeError("No matrix generated")
        N = len(self._matrix)
        M = len(self._matrix[0])
        return self.prompt_template.format(
            N=N,
            M=M,
            matrix="\n".join(" ".join(map(str, row)) for row in self._matrix),
        )

    def _process(self, answer: str | None) -> list[str] | None:
        """Process the answer string into a list of strings."""
        if answer is None:
            return None
        answer = answer.strip()
        try:
            rows = []
            for line in answer.splitlines():
                line = line.strip()
                if line:
                    rows.append(line)
            return rows
        except ValueError:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on the maximum independent set value."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        N = len(self._matrix)
        M = len(self._matrix[0])
        solution = processed_result

        # Check format
        if len(solution) != N or any(len(row) != M for row in solution):
            return 0.0
        if any(c not in "01" for row in solution for c in row):
            return 0.0
        answer_value = 0
        for i in range(N):
            for j in range(M):
                if solution[i][j] == "1":
                    answer_value += self._matrix[i][j]
                    # Check adjacency constraint
                    for di, dj in ((-1, 0), (+1, 0), (0, -1), (0, +1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < N and 0 <= nj < M and solution[ni][nj] == "1":
                            return 0.0
        if answer_value > self._gold_answer + 1e-9:
            raise AssertionError("Answer should not exceed the gold answer")

        # Calculate reward based on strategy
        if self._rewarding_strategy == "(answer/gold)^beta":
            return self._rewarding_weight * (
                (answer_value / self._gold_answer) ** self._rewarding_beta
            )
        elif self._rewarding_strategy == "gold=answer":
            return self._rewarding_weight * (answer_value == self._gold_answer)
        else:
            raise NotImplementedError(
                f"Unknown rewarding strategy: {self._rewarding_strategy}"
            )

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the grid with a beautiful visual representation."""
        if self._matrix is None:
            raise RuntimeError("No matrix generated")

        rows, cols = len(self._matrix), len(self._matrix[0])
        cell_px = self._cell_px
        padding = self._padding

        width = padding * 2 + cols * cell_px
        height = padding * 2 + rows * cell_px

        # Light blue-gray background
        img = Image.new("RGB", (width, height), (240, 245, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.4))
        else:
            font = ImageFont.load_default()

        # Draw cells with gradient coloring based on value
        max_value = max(max(row) for row in self._matrix)
        min_value = min(min(row) for row in self._matrix)

        for r in range(rows):
            for c in range(cols):
                val = self._matrix[r][c]
                # Gradient from light green to deep green based on value
                if max_value > min_value:
                    ratio = (val - min_value) / (max_value - min_value)
                else:
                    ratio = 0.5

                # Light green (220, 240, 220) to medium green (120, 200, 140)
                cell_r = int(220 - ratio * 100)
                cell_g = int(240 - ratio * 40)
                cell_b = int(220 - ratio * 80)
                cell_color = (cell_r, cell_g, cell_b)

                x0 = padding + c * cell_px
                y0 = padding + r * cell_px
                x1 = x0 + cell_px
                y1 = y0 + cell_px

                # Draw filled cell
                draw.rectangle([x0, y0, x1, y1], fill=cell_color)

        # Draw grid lines (darker and thicker for professional look)
        for r in range(rows + 1):
            y = padding + r * cell_px
            draw.line(
                (padding, y, padding + cols * cell_px, y), fill=(50, 60, 70), width=2
            )
        for c in range(cols + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding, x, padding + rows * cell_px), fill=(50, 60, 70), width=2
            )

        # Draw values in cells
        for r in range(rows):
            for c in range(cols):
                val = str(self._matrix[r][c])
                bbox = draw.textbbox((0, 0), val, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2

                # Draw text with shadow for better visibility
                draw.text(
                    (cx - tw // 2 + 1, cy - th // 2 + 1),
                    val,
                    fill=(200, 200, 200),
                    font=font,
                )
                draw.text(
                    (cx - tw // 2, cy - th // 2), val, fill=(20, 30, 40), font=font
                )

        return img

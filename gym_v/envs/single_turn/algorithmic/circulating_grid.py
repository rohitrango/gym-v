"""Circulating grid environment for gym-v (self-contained)."""

from __future__ import annotations

from collections import deque
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class CirculatingGridEnv(Env):
    # Meta: source=RLVE, category=algorithmic, turn=single
    """RLVE Circulating Grid as a single-turn environment.

    Source: https://www.luogu.com.cn/problem/P3965

    Given a grid with directional arrows (L/R/U/D) in each cell, modify the
    minimum number of cells so that starting from any cell, it is possible to
    eventually return to the same cell by following the arrows.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""Consider a {R} × {C} grid, where each cell has coordinates (i, j) (0 ≤ i < {R}, 0 ≤ j < {C}). Each cell contains one of the characters `L`, `R`, `U`, or `D`, meaning:
- `L`: moves to (i, (j - 1) MOD {C})
- `R`: moves to (i, (j + 1) MOD {C})
- `U`: moves to ((i - 1) MOD {R}, j)
- `D`: moves to ((i + 1) MOD {R}, j)
Here, (-1 MOD N) = N - 1.

You are given such a grid:
{grid}

Modify any number of cells so that the resulting grid satisfies the following condition: Starting from any cell, it must be possible to eventually return to the same cell (simply standing there at the beginning does not count). Can you use as small the number of changes (i.e., number of cells modified) as possible? Output the modified grid in the same format — exactly {R} lines, each containing {C} characters (`L`, `R`, `U`, or `D`) with **no separators**."""

    def __init__(
        self,
        max_r_c: int = 5,
        cell_px: int = 56,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._max_r_c = max_r_c
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._grid: list[list[str]] | None = None
        self._R: int | None = None
        self._C: int | None = None
        self._gold_answer: int | None = None
        self._oracle_answer: str | None = None
        self._prompt: str | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        size_hint = f"{self._R} x {self._C}" if self._R and self._C else "R x C"
        return dedent(
            f"""
            This is a grid-based optimization puzzle. Each cell in the grid contains a
            directional arrow (L, R, U, D) that determines movement to an adjacent cell.
            Movement wraps around at the grid edges (toroidal topology). The goal is to
            modify the minimum number of cells so that starting from any cell, following
            the arrows will eventually return to the starting cell.

            Circulating Grid puzzle rules:
            1) Each cell contains a direction: L (left), R (right), U (up), or D (down).
            2) Following the direction from a cell takes you to another cell (with wrapping).
            3) Goal: Modify the minimum number of cells so that starting from ANY cell,
               following the arrows will eventually return to the starting cell.
            4) The circulation condition requires that every cell has exactly one incoming arrow
               (each cell must be reachable by exactly one other cell).

            In the image:
            - Each cell shows a direction arrow: → (R), ← (L), ↑ (U), or ↓ (D)
            - The grid is {size_hint}

            Output Format: Output the modified grid — exactly {self._R or "R"} lines, each
            containing {self._C or "C"} characters (L, R, U, or D) with no separators.
            The optimal solution requires exactly {self._gold_answer} changes.
            """
        ).strip()

    def _get_state_text(self) -> str:
        """Return text representation of the circulating grid."""
        if self._grid is None:
            return ""
        return "\n".join("".join(row) for row in self._grid)

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
        """Generate a random circulating grid puzzle."""
        max_r_c = self._max_r_c
        if max_r_c < 2:
            raise ValueError("max_r_c must be >= 2")

        R = int(self.np_random.integers(2, max_r_c + 1))
        C = int(self.np_random.integers(2, max_r_c + 1))
        self._R = R
        self._C = C

        # Generate random distribution weights for L, R, U, D
        lrud_distribution = [
            int(self.np_random.integers(1, R * C + 1)) for _ in range(4)
        ]

        # Generate random grid
        grid = []
        for _ in range(R):
            row = []
            for _ in range(C):
                choice = self.np_random.choice(
                    ["L", "R", "U", "D"],
                    p=[w / sum(lrud_distribution) for w in lrud_distribution],
                )
                row.append(choice)
            grid.append(row)
        self._grid = grid

        # Compute gold answer using min-cost max-flow
        self._gold_answer, optimal_grid = self._compute_gold_answer(grid, R, C)

        # Set the optimal grid as oracle answer
        self._oracle_answer = "\n".join("".join(row) for row in optimal_grid)

    def _compute_gold_answer(
        self, grid: list[list[str]], R: int, C: int
    ) -> tuple[int, list[list[str]]]:
        """Compute the minimum number of changes needed using min-cost max-flow.

        Returns:
            (min_changes, optimal_grid)
        """
        # Directions: L, R, U, D
        # Order must match loop below
        DIRS = ["L", "R", "U", "D"]
        DX = [0, 0, -1, 1]  # row delta
        DY = [-1, 1, 0, 0]  # col delta
        DIR_ID = {"L": 0, "R": 1, "U": 2, "D": 3}

        class Edge:
            __slots__ = ("to", "rev", "cap", "cost", "is_real")

            def __init__(self, to, rev, cap, cost, is_real=False):
                self.to = to
                self.rev = rev
                self.cap = cap
                self.cost = cost
                self.is_real = (
                    is_real  # True if this is a u->v edge representing a move
                )

        def add_edge(graph, u, v, cap, cost, is_real=False):
            graph[u].append(Edge(v, len(graph[v]), cap, cost, is_real))
            graph[v].append(Edge(u, len(graph[u]) - 1, 0, -cost, False))

        def min_cost_max_flow(graph, N, s, t, INF):
            flow = 0
            cost = 0
            dist = [0] * N
            inq = [False] * N
            prev_node = [-1] * N
            prev_edge = [-1] * N

            while True:
                # SPFA to find shortest augmenting path by cost
                for i in range(N):
                    dist[i] = INF
                    inq[i] = False
                    prev_node[i] = -1
                    prev_edge[i] = -1
                dist[s] = 0
                q = deque([s])
                inq[s] = True

                while q:
                    u = q.popleft()
                    inq[u] = False
                    for ei, e in enumerate(graph[u]):
                        if e.cap > 0:
                            v = e.to
                            nd = dist[u] + e.cost
                            if nd < dist[v]:
                                dist[v] = nd
                                prev_node[v] = u
                                prev_edge[v] = ei
                                if not inq[v]:
                                    inq[v] = True
                                    q.append(v)

                if prev_node[t] == -1:
                    break  # no more augmenting paths

                # Find bottleneck
                addf = INF
                v = t
                while v != s:
                    u = prev_node[v]
                    ei = prev_edge[v]
                    e = graph[u][ei]
                    if e.cap < addf:
                        addf = e.cap
                    v = u

                # Augment
                v = t
                while v != s:
                    u = prev_node[v]
                    ei = prev_edge[v]
                    e = graph[u][ei]
                    e.cap -= addf
                    graph[v][e.rev].cap += addf
                    cost += addf * e.cost
                    v = u

                flow += addf

            return flow, cost

        # MP holds the direction id (0..3) for each cell
        MP = [[0] * C for _ in range(R)]
        for i in range(R):
            for j in range(C):
                MP[i][j] = DIR_ID[grid[i][j]]

        n_left = R * C
        offset = n_left
        s = 2 * n_left
        t = s + 1
        N = t + 1

        # INF derived from input size; safely larger than any possible path cost
        INF = R * C * 4 + 5

        graph = [[] for _ in range(N)]

        # Build edges from each cell (left partition) to its 4 neighbors (right partition)
        # Store mapping to retrieve direction later
        # map (u, edge_index_in_graph[u]) -> direction_index (0..3)
        edge_dir_map = {}

        for i in range(R):
            for j in range(C):
                u = i * C + j
                for k in range(4):
                    ni = (i + DX[k]) % R
                    nj = (j + DY[k]) % C
                    v = offset + (ni * C + nj)
                    cost = 0 if k == MP[i][j] else 1

                    edge_idx = len(graph[u])
                    edge_dir_map[(u, edge_idx)] = k
                    add_edge(graph, u, v, 1, cost, is_real=True)

        # Source to all left nodes; all right nodes to sink
        for u in range(n_left):
            add_edge(graph, s, u, 1, 0)
        for v in range(offset, offset + n_left):
            add_edge(graph, v, t, 1, 0)

        _, total_cost = min_cost_max_flow(graph, N, s, t, INF)

        # Reconstruct the optimal grid
        optimal_grid = [[""] * C for _ in range(R)]

        for i in range(R):
            for j in range(C):
                u = i * C + j
                # Find the outgoing edge from u that has flow (cap == 0 because capacity was 1)
                for ei, e in enumerate(graph[u]):
                    if e.is_real and e.cap == 0:
                        # This edge was used in the flow
                        k = edge_dir_map.get((u, ei))
                        if k is not None:
                            optimal_grid[i][j] = DIRS[k]
                            break

                # Fallback (should not happen if max flow is perfect)
                if optimal_grid[i][j] == "":
                    optimal_grid[i][j] = grid[i][
                        j
                    ]  # Keep original if no flow (unexpected)

        return total_cost, optimal_grid

    def _prompt_generate(self) -> str:
        if self._grid is None:
            raise RuntimeError("No grid generated")
        return self.prompt_template.format(
            R=self._R,
            C=self._C,
            grid="\n".join("".join(row) for row in self._grid),
        )

    def _process(self, answer: str | None) -> list[str] | None:
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
        """Score the answer based on validity and optimality."""
        processed_result = self._process(answer)
        if processed_result is None:
            return 0.0
        grid = processed_result

        # Check format
        if len(grid) != self._R:
            return 0.0
        if not all(len(row) == self._C for row in grid):
            return 0.0
        if not all(all(c in "LRUD" for c in row) for row in grid):
            return 0.0
        in_degree = [[0] * self._C for _ in range(self._R)]
        for i in range(self._R):
            for j in range(self._C):
                if grid[i][j] == "L":
                    in_degree[i][(j - 1 + self._C) % self._C] += 1
                elif grid[i][j] == "R":
                    in_degree[i][(j + 1) % self._C] += 1
                elif grid[i][j] == "U":
                    in_degree[(i - 1 + self._R) % self._R][j] += 1
                elif grid[i][j] == "D":
                    in_degree[(i + 1) % self._R][j] += 1

        # Each cell must have exactly one incoming arrow
        if not all(
            in_degree[i][j] == 1 for i in range(self._R) for j in range(self._C)
        ):
            return 0.0
        answer_changes = sum(
            int(grid[i][j] != self._grid[i][j])
            for i in range(self._R)
            for j in range(self._C)
        )
        gold = self._gold_answer

        # Reward based on optimality
        if answer_changes == 0:
            if gold == 0:
                return 1.0
            else:
                # Valid but not optimal
                return (gold / answer_changes) ** 5 if answer_changes > 0 else 0.0

        return (gold / answer_changes) ** 5

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the grid with beautiful directional arrows."""
        if self._grid is None:
            raise RuntimeError("No grid generated")

        rows, cols = self._R, self._C
        cell_px = self._cell_px
        padding = self._padding

        width = padding * 2 + cols * cell_px
        height = padding * 2 + rows * cell_px
        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font = ImageFont.truetype(font_path, int(cell_px * 0.6))
        else:
            font = ImageFont.load_default()

        # Draw grid lines
        for r in range(rows + 1):
            y = padding + r * cell_px
            draw.line(
                (padding, y, padding + cols * cell_px, y),
                fill=(30, 30, 30),
                width=2,
            )
        for c in range(cols + 1):
            x = padding + c * cell_px
            draw.line(
                (x, padding, x, padding + rows * cell_px),
                fill=(30, 30, 30),
                width=2,
            )

        # Draw directional arrows
        arrow_map = {
            "L": "←",
            "R": "→",
            "U": "↑",
            "D": "↓",
        }

        for r in range(rows):
            for c in range(cols):
                direction = self._grid[r][c]
                arrow = arrow_map[direction]
                bbox = draw.textbbox((0, 0), arrow, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = padding + c * cell_px + cell_px // 2
                cy = padding + r * cell_px + cell_px // 2
                draw.text(
                    (cx - tw // 2, cy - th // 2),
                    arrow,
                    fill=(10, 10, 200),
                    font=font,
                )

        return img

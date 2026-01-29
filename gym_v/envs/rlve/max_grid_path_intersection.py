"""Max Grid Path Intersection environment for gym-v (self-contained)."""

from __future__ import annotations

from collections import deque
from importlib import resources
from textwrap import dedent
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from gym_v import Env, Observation
from gym_v.logger import get_logger

logger = get_logger()


class RLVEMaxGridPathIntersectionEnv(Env):
    """RLVE Max Grid Path Intersection as a single-turn environment.

    Given an N × N grid, find the maximum total sum that can be collected
    by traversing from (0, 0) to (N-1, N-1) exactly K times. Each path can
    only move right or down. When stepping on a cell, collect its value and
    set it to 0 for future paths.
    """

    assets_dir = resources.files("gym_v.envs") / "assets"

    prompt_template = r"""You are given an {N} × {N} grid (0-indexed) of non-negative integers (given in **row-major order**):
{grid}

You will start at cell (0, 0) and move to cell ({N_minus_1}, {N_minus_1}) exactly {K} times. Each time, you can only move **right** or **down** at each step. When you step on a cell during a path, you collect its value and set it to 0 (so future paths will see it as 0). Your goal is to **maximize the total sum** collected across all {K} paths.

**Output Format:** A single integer — the maximum total sum that can be collected after {K} such paths."""

    def __init__(
        self,
        n: int = 5,
        cell_px: int = 70,
        padding: int = 24,
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._n_param = n
        self._cell_px = cell_px
        self._padding = padding
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        if n < 3:
            raise ValueError("n should be >= 3")

        self._n: int | None = None
        self._k: int | None = None
        self._grid: list[list[int]] | None = None
        self._prompt: str | None = None
        self._oracle_answer: int | None = None
        self._last_image: Image.Image | None = None

    @property
    def description(self) -> str:
        if self._n and self._k:
            path_word = "path" if self._k == 1 else "paths"
            size_hint = f"{self._n} × {self._n} grid, {self._k} {path_word}"
        else:
            size_hint = "N × N grid, K paths"

        return dedent(
            f"""
            Max Grid Path Intersection Problem:

            Given a {size_hint} of non-negative integers, find the maximum total sum
            that can be collected across K paths from top-left (0, 0) to bottom-right (N-1, N-1).

            Rules:
            1) Each path can only move RIGHT or DOWN at each step
            2) When a cell is visited, collect its value and set it to 0
            3) Future paths will see previously visited cells as 0
            4) All K paths must go from (0, 0) to (N-1, N-1)
            5) Goal: Maximize the total sum collected across all K paths

            In the visualization:
            - Grid cells show their initial values
            - Cell colors indicate value magnitude (warmer = higher value)
            - Multiple optimal paths may exist
            - The paths can share cells (but only the first path gets the value)
            - Paths are shown with different colors
            - Intersection points are highlighted where paths overlap

            Output format: A single integer (maximum total sum).
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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
            },
        )
        info = {
            "oracle_answer": str(self._oracle_answer),
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
            text=state_text,
            metadata={
                "text_prompt": f"{state_text}\n\n{self.description}",
            },
        )
        info = {
            "oracle_answer": str(self._oracle_answer),
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
        """Generate a max grid path intersection problem instance.

        Ports generation logic from RLVE using self.np_random.
        """
        N = self._n_param
        if N < 3:
            raise ValueError("N should be greater than or equal to 3")

        K = int(self.np_random.integers(1, N // 2 + 1))

        A = [
            [int(self.np_random.integers(0, N + 1)) for _ in range(N)] for _ in range(N)
        ]

        def max_cost_flow(N: int, K: int, A: list[list[int]]) -> int:
            """Compute maximum cost flow for the grid path problem."""
            # Number of nodes: each cell has in-node and out-node
            total_nodes = 2 * N * N
            # Adjacency list: each entry is [to, capacity, cost, rev]
            ADJ = [[] for _ in range(total_nodes)]

            def add_edge(u: int, v: int, cap: int, cost: int) -> None:
                """Add edge to flow network."""
                # forward edge
                forward = [v, cap, cost, None]
                # reverse edge
                backward = [u, 0, -cost, None]
                # link edges for capacity updates
                forward[3] = backward
                backward[3] = forward
                ADJ[u].append(forward)
                ADJ[v].append(backward)

            def node_id(i: int, j: int, is_out: bool) -> int:
                """Get node ID for cell (i, j)."""
                # 0-indexed: cells at (i, j) share indices 0..N*N-1 for in-nodes,
                # N*N..2*N*N-1 for out-nodes
                base = N * N if is_out else 0
                return base + i * N + j

            # Build the flow network
            for i in range(N):
                for j in range(N):
                    in_id = node_id(i, j, False)
                    out_id = node_id(i, j, True)
                    # Pick the cell's value on one of the K visits
                    add_edge(in_id, out_id, 1, A[i][j])  # one with reward
                    add_edge(in_id, out_id, K - 1, 0)  # others free
                    # Move right or down (up to K walkers)
                    if j + 1 < N:
                        add_edge(out_id, node_id(i, j + 1, False), K, 0)
                    if i + 1 < N:
                        add_edge(out_id, node_id(i + 1, j, False), K, 0)

            s = node_id(0, 0, False)
            t = node_id(N - 1, N - 1, True)
            total_cost = 0

            # If K is zero, there is no flow and cost is zero
            if K == 0:
                return 0

            # Successive SPFA for maximum-cost flow
            while True:
                DIST = [float("-inf")] * total_nodes
                FLOW = [0] * total_nodes
                INQUEUE = [False] * total_nodes
                PREV_NODE = [None] * total_nodes
                PREV_EDGE = [None] * total_nodes

                queue = deque([s])
                DIST[s] = 0
                FLOW[s] = K  # maximum possible augment per iteration
                INQUEUE[s] = True

                # Find longest path from s to t in residual graph
                while queue:
                    u = queue.popleft()
                    INQUEUE[u] = False
                    for edge in ADJ[u]:
                        v, cap, cost, rev = edge
                        if cap > 0 and DIST[v] < DIST[u] + cost:
                            DIST[v] = DIST[u] + cost
                            FLOW[v] = min(FLOW[u], cap)
                            PREV_NODE[v] = u
                            PREV_EDGE[v] = edge
                            if not INQUEUE[v]:
                                queue.append(v)
                                INQUEUE[v] = True

                # If there's no augmenting path, we're done
                if DIST[t] == float("-inf"):
                    break

                # Augment along the path
                f = FLOW[t]
                total_cost += f * DIST[t]
                v = t
                while v != s:
                    edge = PREV_EDGE[v]
                    # reduce forward capacity
                    edge[1] -= f
                    # increase reverse capacity
                    edge[3][1] += f
                    v = PREV_NODE[v]

            return total_cost

        self._n = N
        self._k = K
        self._grid = A
        self._oracle_answer = max_cost_flow(N, K, A)

    def _prompt_generate(self) -> str:
        """Generate the prompt text for the problem."""
        if self._n is None:
            raise RuntimeError("No problem generated")
        return self.prompt_template.format(
            N=self._n,
            N_minus_1=self._n - 1,
            K=self._k,
            grid="\n".join(" ".join(map(str, row)) for row in self._grid),
        )

    def _process(self, answer: str | None) -> int | None:
        """Process the answer string into an integer."""
        if answer is not None:
            answer = answer.strip()
            try:
                int_answer = int(answer)
                return int_answer
            except ValueError:
                return None
        else:
            return None

    def _score_answer(self, answer: str) -> float:
        """Score the answer based on correctness.

        Returns:
            -1.0: wrong format (not an integer)
             0.0: wrong answer
            +1.0: correct answer
        """
        processed_result = self._process(answer)
        if processed_result is not None:
            if processed_result == self._oracle_answer:
                return 1.0
            else:
                return 0.0
        else:
            return 0.0

    def render(self) -> Image.Image | list[Image.Image] | None:
        """Render the grid with path visualization.

        Shows:
        - Grid with cell values
        - Color-coded cells based on value magnitude
        - Visual representation of optimal paths
        - Intersection highlights
        """
        if self._n is None or self._grid is None:
            raise RuntimeError("No problem generated")

        cell_px = self._cell_px
        padding = self._padding
        n = self._n
        k = self._k

        # Calculate dimensions
        grid_width = n * cell_px
        grid_height = n * cell_px

        # Space for title and legend
        title_height = 70
        legend_height = 80

        width = padding * 2 + grid_width
        height = padding * 3 + grid_height + title_height + legend_height

        img = Image.new("RGB", (width, height), (250, 250, 250))
        draw = ImageDraw.Draw(img)

        # Load font
        font_path = None
        font_path_candidate = self.assets_dir / "DejaVuSans.ttf"
        if font_path_candidate.exists():
            font_path = str(font_path_candidate)

        if font_path:
            font_large = ImageFont.truetype(font_path, int(cell_px * 0.35))
            font_title = ImageFont.truetype(font_path, 28)
            font_medium = ImageFont.truetype(font_path, 18)
            font_small = ImageFont.truetype(font_path, 14)
        else:
            font_large = ImageFont.load_default()
            font_title = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw title
        title = f"Max Grid Path Intersection - {k} Paths on {n}×{n} Grid"
        title_bbox = draw.textbbox((0, 0), title, font=font_title)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (width - title_width) // 2
        draw.text((title_x, 15), title, fill=(30, 30, 30), font=font_title)

        # Find max value for color scaling
        max_val = max(max(row) for row in self._grid)
        if max_val == 0:
            max_val = 1  # Avoid division by zero

        # Draw grid
        grid_x = padding
        grid_y = padding + title_height

        # Draw grid cells with color intensity based on value
        for r in range(n):
            for c in range(n):
                x = grid_x + c * cell_px
                y = grid_y + r * cell_px

                value = self._grid[r][c]
                # Color gradient from white (0) to orange-red (max)
                intensity_ratio = value / max_val

                # Gradient: white -> light yellow -> orange -> red
                if intensity_ratio < 0.33:
                    t = intensity_ratio / 0.33
                    red = 255
                    green = 255
                    blue = int(255 - t * 100)
                elif intensity_ratio < 0.66:
                    t = (intensity_ratio - 0.33) / 0.33
                    red = 255
                    green = int(255 - t * 90)
                    blue = int(155 - t * 55)
                else:
                    t = (intensity_ratio - 0.66) / 0.34
                    red = int(255 - t * 35)
                    green = int(165 - t * 100)
                    blue = int(100 - t * 100)

                cell_color = (red, green, blue)

                draw.rectangle(
                    [x + 2, y + 2, x + cell_px - 2, y + cell_px - 2],
                    fill=cell_color,
                    outline=None,
                )

        # Highlight start and end cells
        start_x = grid_x
        start_y = grid_y
        draw.rectangle(
            [start_x + 2, start_y + 2, start_x + cell_px - 2, start_y + cell_px - 2],
            outline=(50, 200, 50),
            width=5,
        )

        end_x = grid_x + (n - 1) * cell_px
        end_y = grid_y + (n - 1) * cell_px
        draw.rectangle(
            [end_x + 2, end_y + 2, end_x + cell_px - 2, end_y + cell_px - 2],
            outline=(200, 50, 50),
            width=5,
        )

        # Draw sample paths (illustrative, not the actual optimal paths)
        # Show K different colored paths as examples
        path_colors = [
            (70, 130, 180),  # Steel blue
            (218, 112, 214),  # Orchid
            (255, 165, 0),  # Orange
            (46, 139, 87),  # Sea green
            (220, 20, 60),  # Crimson
        ]

        # Generate K simple paths for visualization
        for path_idx in range(min(k, 3)):  # Show up to 3 paths for clarity
            color = path_colors[path_idx % len(path_colors)]
            # Simple greedy path favoring high values
            r, c = 0, 0
            path_cells = [(r, c)]

            while r < n - 1 or c < n - 1:
                # Choose direction based on which has higher value
                if r == n - 1:
                    c += 1
                elif c == n - 1:
                    r += 1
                else:
                    # Pick direction with higher value
                    right_val = self._grid[r][c + 1]
                    down_val = self._grid[r + 1][c]
                    if right_val > down_val:
                        c += 1
                    elif down_val > right_val:
                        r += 1
                    else:
                        # Alternate or use path index to differentiate
                        if path_idx % 2 == 0:
                            c += 1
                        else:
                            r += 1
                path_cells.append((r, c))

            # Draw path arrows
            for i in range(len(path_cells) - 1):
                r1, c1 = path_cells[i]
                r2, c2 = path_cells[i + 1]

                cx1 = grid_x + c1 * cell_px + cell_px // 2
                cy1 = grid_y + r1 * cell_px + cell_px // 2
                cx2 = grid_x + c2 * cell_px + cell_px // 2
                cy2 = grid_y + r2 * cell_px + cell_px // 2

                # Offset paths slightly so they don't overlap completely
                offset = (path_idx - 1) * 6
                if r1 == r2:  # Horizontal movement
                    cy1 += offset
                    cy2 += offset
                else:  # Vertical movement
                    cx1 += offset
                    cx2 += offset

                # Draw path line
                draw.line(
                    [(cx1, cy1), (cx2, cy2)],
                    fill=color,
                    width=4,
                )

                # Draw arrowhead
                if c2 > c1:  # Right arrow
                    draw.polygon(
                        [(cx2 - 8, cy2 - 4), (cx2, cy2), (cx2 - 8, cy2 + 4)],
                        fill=color,
                    )
                else:  # Down arrow
                    draw.polygon(
                        [(cx2 - 4, cy2 - 8), (cx2, cy2), (cx2 + 4, cy2 - 8)],
                        fill=color,
                    )

        # Draw grid lines
        for r in range(n + 1):
            y = grid_y + r * cell_px
            draw.line((grid_x, y, grid_x + grid_width, y), fill=(80, 80, 80), width=2)
        for c in range(n + 1):
            x = grid_x + c * cell_px
            draw.line((x, grid_y, x, grid_y + grid_height), fill=(80, 80, 80), width=2)

        # Draw cell values
        for r in range(n):
            for c in range(n):
                v = str(self._grid[r][c])
                bbox = draw.textbbox((0, 0), v, font=font_large)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                cx = grid_x + c * cell_px + cell_px // 2
                cy = grid_y + r * cell_px + cell_px // 2

                # Use contrasting text color
                value = self._grid[r][c]
                intensity_ratio = value / max_val
                text_color = (0, 0, 0) if intensity_ratio < 0.5 else (255, 255, 255)

                draw.text(
                    (cx - tw // 2, cy - th // 2), v, fill=text_color, font=font_large
                )

        # Draw legend
        legend_y = grid_y + grid_height + padding * 2

        # Start and End indicators
        legend_items = [
            ("START (0, 0)", (50, 200, 50)),
            (f"END ({n - 1}, {n - 1})", (200, 50, 50)),
        ]

        legend_x = padding
        for label, color in legend_items:
            # Draw color box
            draw.rectangle(
                [legend_x, legend_y, legend_x + 20, legend_y + 20],
                outline=color,
                width=4,
            )
            # Draw label
            draw.text(
                (legend_x + 30, legend_y + 2),
                label,
                fill=(30, 30, 30),
                font=font_medium,
            )
            legend_x += 200

        legend_y += 35

        # Path info
        path_info = f"Find {k} optimal paths that maximize the total collected sum"
        draw.text((padding, legend_y), path_info, fill=(100, 100, 100), font=font_small)

        return img
